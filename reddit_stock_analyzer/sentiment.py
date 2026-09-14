"""Post-level sentiment via FinTwitBERT.

``transformers`` and ``torch`` are optional extras: without them every post is
labelled ``neutral`` and the rest of the pipeline (scraping, ticker
recognition, trend maths) still works. Install with::

    pip install "reddit-stock-analyzer[sentiment]"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from .models import Sentiment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "StephanAkkerman/FinTwitBERT-wsb-sentiment"

#: FinTwitBERT emits BULLISH/BEARISH/NEUTRAL; the generic LABEL_n names are
#: the fallback for checkpoints published without an id2label mapping.
_LABEL_MAP: dict[str, Sentiment] = {
    "BULLISH": "bullish",
    "POSITIVE": "bullish",
    "LABEL_2": "bullish",
    "BEARISH": "bearish",
    "NEGATIVE": "bearish",
    "LABEL_0": "bearish",
    "NEUTRAL": "neutral",
    "LABEL_1": "neutral",
}

#: BERT truncates anyway; trimming first keeps tokenisation cheap on the
#: 10k-character DD posts r/wallstreetbets is fond of.
MAX_CHARS = 2000


def normalize_label(label: str) -> Sentiment:
    """Map a model label onto ``bullish`` / ``bearish`` / ``neutral``."""
    return _LABEL_MAP.get(str(label).strip().upper(), "neutral")


def signed_score(label: Sentiment, confidence: float) -> float:
    """Return confidence signed by direction: +bullish, -bearish, 0 neutral."""
    if label == "bullish":
        return float(confidence)
    if label == "bearish":
        return -float(confidence)
    return 0.0


class SentimentAnalyzer:
    """Classify financial social-media text as bullish/bearish/neutral.

    Parameters
    ----------
    model_path : str
        Hugging Face model id or local path.
    pipeline : callable, optional
        Pre-built ``transformers`` pipeline (or any callable taking a list of
        strings and returning ``[{"label": ..., "score": ...}, ...]``). Mainly
        for tests and for sharing one loaded model.
    batch_size : int, default 16
        Texts per forward pass.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        pipeline: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_path = model_path
        self._pipe = pipeline
        self._load_failed = False
        self.batch_size = max(1, int(batch_size))

    @property
    def available(self) -> bool:
        """Whether classification can run (``False`` means neutral-only)."""
        if self._pipe is not None:
            return True
        if self._load_failed:
            return False
        return self._load() is not None

    def _load(self) -> Any | None:
        if self._pipe is not None or self._load_failed:
            return self._pipe
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "transformers is not installed; sentiment falls back to neutral. "
                'Install with: pip install "reddit-stock-analyzer[sentiment]"'
            )
            self._load_failed = True
            return None

        try:
            logger.info("Loading sentiment model %s...", self.model_path)
            self._pipe = hf_pipeline(
                "text-classification",
                model=self.model_path,
                truncation=True,
                max_length=512,
            )
        except Exception:
            logger.warning(
                "Could not load sentiment model %s; falling back to neutral.",
                self.model_path,
                exc_info=True,
            )
            self._load_failed = True
            return None
        return self._pipe

    def analyze_batch(self, texts: Sequence[str]) -> list[tuple[Sentiment, float]]:
        """Classify *texts*.

        Returns
        -------
        list of (label, confidence)
            One entry per input, in order. ``("neutral", 0.0)`` for empty
            strings and whenever the model is unavailable.
        """
        if not texts:
            return []

        neutral: tuple[Sentiment, float] = ("neutral", 0.0)
        pipe = self._load()
        if pipe is None:
            return [neutral] * len(texts)

        # Keep the index mapping so blank inputs never reach the model but
        # still occupy their slot in the result.
        indexed = [
            (i, text.strip()[:MAX_CHARS])
            for i, text in enumerate(texts)
            if text and text.strip()
        ]
        results: list[tuple[Sentiment, float]] = [neutral] * len(texts)
        if not indexed:
            return results

        try:
            predictions = pipe(
                [text for _, text in indexed], batch_size=self.batch_size
            )
        except Exception:
            logger.warning(
                "Sentiment inference failed; returning neutral.", exc_info=True
            )
            return results

        for (index, _), prediction in zip(indexed, predictions or ()):
            if isinstance(prediction, list):  # top_k pipelines return a list
                prediction = prediction[0] if prediction else {}
            label = normalize_label((prediction or {}).get("label", ""))
            confidence = float((prediction or {}).get("score", 0.0) or 0.0)
            results[index] = (label, confidence)
        return results

    async def analyze_batch_async(
        self, texts: Sequence[str]
    ) -> list[tuple[Sentiment, float]]:
        """Off-thread :meth:`analyze_batch`, so an event loop keeps serving."""
        if not texts:
            return []
        return await asyncio.to_thread(self.analyze_batch, texts)

    def warm_up(self) -> None:
        """Load the model eagerly (e.g. during app startup)."""
        self.analyze_batch(["warm up"])
