from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SemanticEvaluator:
    def __init__(self):
        self.model_name = "TEN-framework/TEN_Turn_Detection"
        self.cache_dir = "./models"
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

    def evaluate(self, text: str) -> float:
        """
        Evaluate if the speaker is likely to continue talking.

        Args:
            text: The input sentence/utterance

        Returns:
            float: Probability that the speaker will continue talking (0.0 to 1.0)
        """

        # tokenize input
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Assuming the model returns [prob_finished, prob_continue]
            # We want the probability of continuing
            continue_prob = probabilities[0][1].item()

        return continue_prob


def main():
    print("starting")

    evaluator = SemanticEvaluator()

    test_cases = [
        "How can I help you today?",
        "Well, I was looking into the available rental cars on your website. And, umm...",
    ]

    for text in test_cases:
        prob = evaluator.evaluate(text)
        print(f"Text: '{text}'")
        print(f"Continue probability: {prob:.3f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
