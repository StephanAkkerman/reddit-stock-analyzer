"""Ticker recognition, backed by the ``stock-recognizer`` package.

`stock-recognizer <https://github.com/StephanAkkerman/stock-recognizer>`_ does
the hard part — cashtags, market-data validation, company-name mapping and a
fine-tuned GLiNER2 model for "TSMC" → ``TSM``. This module is the thin
adapter: lazy construction (loading market data and the AI model costs
seconds and, with ``use_ai``, hundreds of MB), a process-wide default
instance, and graceful degradation when the optional AI stack is missing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsRecognize(Protocol):
    """The slice of ``stock_recognizer.StockRecognizer`` this package uses."""

    def recognize(self, text: str) -> list[str]: ...


class TickerExtractor:
    """Extract stock tickers from post text.

    Parameters
    ----------
    use_ai : bool, default True
        Use ``StockRecognizer.recognize_ai`` (GLiNER2 + regex + market data).
        ``False`` keeps the regex/market-data path only: no torch, no model
        download, a few milliseconds per call instead of tens.
    recognizer : SupportsRecognize, optional
        Pre-built recognizer to use instead of constructing one. Mainly for
        tests and for sharing a single loaded model across components.
    blocklist : iterable of str, optional
        Extra symbols to drop on top of ``stock-recognizer``'s own ambiguity
        filtering — handy for subreddit-specific noise.
    recognizer_kwargs : dict, optional
        Extra keyword arguments forwarded to ``StockRecognizer``, e.g.
        ``{"include_global_majors": True}``.
    """

    def __init__(
        self,
        *,
        use_ai: bool = True,
        recognizer: SupportsRecognize | None = None,
        blocklist: Iterable[str] | None = None,
        recognizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.use_ai = use_ai
        self._recognizer = recognizer
        self._recognizer_kwargs = dict(recognizer_kwargs or {})
        self.blocklist = {s.upper() for s in (blocklist or ())}

    @property
    def recognizer(self) -> SupportsRecognize:
        """The underlying recognizer, constructed on first access."""
        if self._recognizer is None:
            self._recognizer = self._build()
        return self._recognizer

    def _build(self) -> SupportsRecognize:
        try:
            from stock_recognizer import StockRecognizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "stock-recognizer is required for ticker extraction. "
                "Install it with `pip install stock-recognizer`."
            ) from exc

        logger.info("Loading StockRecognizer (use_ai=%s)...", self.use_ai)
        try:
            return StockRecognizer(use_ai=self.use_ai, **self._recognizer_kwargs)
        except Exception:
            if not self.use_ai:
                raise
            # The AI path needs torch/gliner2 and a model download; neither is
            # available in every deployment. The regex + market-data path still
            # catches cashtags, which is most of Reddit's ticker traffic.
            logger.warning(
                "Could not load StockRecognizer with use_ai=True; "
                "falling back to the regex/market-data path.",
                exc_info=True,
            )
            self.use_ai = False
            return StockRecognizer(use_ai=False, **self._recognizer_kwargs)

    def extract(self, text: str) -> list[str]:
        """Return the distinct tickers mentioned in *text*, sorted.

        Returns
        -------
        list of str
            Uppercase symbols. Empty for empty text or on recognizer failure —
            a single malformed post must never take down a scrape.
        """
        if not text or not text.strip():
            return []

        recognizer = self.recognizer
        try:
            if self.use_ai and hasattr(recognizer, "recognize_ai"):
                symbols = recognizer.recognize_ai(text)
            else:
                symbols = recognizer.recognize(text)
        except Exception:
            logger.warning("Ticker extraction failed for a post", exc_info=True)
            return []

        return sorted(
            {
                symbol.upper()
                for symbol in symbols or ()
                if symbol and symbol.upper() not in self.blocklist
            }
        )

    def extract_many(self, texts: Sequence[str]) -> list[list[str]]:
        """Run :meth:`extract` over *texts*, preserving order."""
        return [self.extract(text) for text in texts]

    async def extract_many_async(self, texts: Sequence[str]) -> list[list[str]]:
        """Off-thread :meth:`extract_many`, so an event loop keeps serving.

        The recognizer is synchronous and CPU-bound; running it inline in an
        async worker would stall every other request for the duration.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self.extract_many, texts)

    def warm_up(self) -> None:
        """Build the recognizer eagerly (e.g. during app startup)."""
        self.extract("$AAPL")


_DEFAULT: TickerExtractor | None = None


def get_default_extractor(*, use_ai: bool = True) -> TickerExtractor:
    """Return a process-wide :class:`TickerExtractor`, building it once.

    Loading market data and the AI adapter is expensive, so callers that make
    many short-lived scrapes should share one instance. The cached instance is
    reused regardless of *use_ai* once created; pass a fresh
    :class:`TickerExtractor` if you need both modes side by side.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TickerExtractor(use_ai=use_ai)
    return _DEFAULT


def reset_default_extractor() -> None:
    """Drop the cached default extractor (used by tests)."""
    global _DEFAULT
    _DEFAULT = None
