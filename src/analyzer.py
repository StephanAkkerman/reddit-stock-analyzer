from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self, model_path: str = "./fintwitbert-wsb-finetuned"):
        # We load the local model we just trained
        self.pipe = pipeline(
            "text-classification", model=model_path, truncation=True, max_length=512
        )

    def analyze_batch(self, texts: list[str]):
        """Returns labels and scores for a list of strings."""
        results = self.pipe(texts)
        return results  # [{'label': 'BULLISH', 'score': 0.99}, ...]
