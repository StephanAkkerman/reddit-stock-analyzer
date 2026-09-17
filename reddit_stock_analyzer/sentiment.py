"""Post sentiment via FinTwitBERT-wsb.

The default model is
`StephanAkkerman/FinTwitBERT-wsb-sentiment <https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment>`_
— BERT pre-trained on financial tweets and fine-tuned on WallStreetBets-style
text, so it reads "puts printing" and "she's gonna rip" the way a trader does
rather than the way a general-purpose sentiment model does.

``transformers`` and ``torch`` arrive transitively with ``stock-recognizer``,
so sentiment works out of the box. If they are somehow absent every post is
labelled ``neutral`` and the rest of the pipeline (scraping, ticker
recognition, trend maths) still works.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Mapping, Sequence

from .models import Sentiment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "StephanAkkerman/FinTwitBERT-wsb-sentiment"

#: Label names financial classifiers publish. FinTwitBERT emits
#: BULLISH/BEARISH/NEUTRAL; the rest are here so a swapped-in model works too.
_NAMED_LABELS: dict[str, Sentiment] = {
    "BULLISH": "bullish",
    "POSITIVE": "bullish",
    "POS": "bullish",
    "BEARISH": "bearish",
    "NEGATIVE": "bearish",
    "NEG": "bearish",
    "NEUTRAL": "neutral",
    "NEU": "neutral",
}

#: A checkpoint published without an ``id2label`` mapping reports LABEL_0,
#: LABEL_1... There is no safe way to guess which index means bullish, and
#: guessing wrong silently inverts every score, so these map to neutral and
#: warn. Pass ``label_map={"LABEL_0": "bearish", ...}`` to state the order.
_GENERIC_LABEL_RE = re.compile(r"^LABEL_\d+$")

#: BERT truncates at 512 tokens anyway; trimming first keeps tokenisation
#: cheap on the 10k-character DD posts r/wallstreetbets is fond of.
MAX_CHARS = 2000


def normalize_label(
    label: str, label_map: Mapping[str, Sentiment] | None = None
) -> Sentiment:
    """Map a model label onto ``bullish`` / ``bearish`` / ``neutral``.

    Parameters
    ----------
    label : str
        Raw label from the classifier.
    label_map : mapping, optional
        Extra or overriding label names, upper-cased keys.

    Returns
    -------
    Sentiment
        ``neutral`` for anything unrecognised.
    """
    key = str(label).strip().upper()
    if label_map and key in label_map:
        return label_map[key]
    return _NAMED_LABELS.get(key, "neutral")


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
    label_map : mapping, optional
        Label name to sentiment, for a model whose labels this package does
        not already know — including one that reports generic ``LABEL_0`` /
        ``LABEL_1`` names, where guessing the order would risk silently
        swapping bullish and bearish.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        pipeline: Any | None = None,
        batch_size: int = 16,
        label_map: Mapping[str, Sentiment] | None = None,
    ) -> None:
        self.model_path = model_path
        self._pipe = pipeline
        self._load_failed = False
        self.batch_size = max(1, int(batch_size))
        self.label_map = {
            str(k).strip().upper(): v for k, v in (label_map or {}).items()
        }
        self._warned_generic = False

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
            raw_label = (prediction or {}).get("label", "")
            label = normalize_label(raw_label, self.label_map)
            confidence = float((prediction or {}).get("score", 0.0) or 0.0)
            if label == "neutral":
                self._warn_if_unmapped(raw_label)
            results[index] = (label, confidence)
        return results

    def _warn_if_unmapped(self, raw_label: str) -> None:
        """Warn once when the model's labels carry no direction.

        A checkpoint without an ``id2label`` mapping reports LABEL_0/LABEL_1,
        which this package refuses to guess at: everything reads neutral until
        a ``label_map`` says which index is which. Silence here would look
        exactly like a genuinely undecided subreddit.
        """
        if self._warned_generic:
            return
        key = str(raw_label).strip().upper()
        if _GENERIC_LABEL_RE.match(key):
            self._warned_generic = True
            logger.warning(
                "Sentiment model %s reports generic labels (%s), so every post "
                "reads neutral. Pass label_map={'LABEL_0': 'bearish', ...} to "
                "map them.",
                self.model_path,
                raw_label,
            )

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
