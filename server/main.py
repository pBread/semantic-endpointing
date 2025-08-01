from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Set seed for reproducible demo results
# Note: TEN-framework/TEN_Turn_Detection should be properly trained for production use
torch.manual_seed(999)


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
        Args:
            text: The input sentence/utterance

        Returns:
            float: Probability that the speaker is finished speaking (0 - 1)
        """
        # Tokenize the input
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # get probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Check how many classes we have
            num_classes = probabilities.shape[-1]

            if num_classes == 2:
                prob_finished = probabilities[0][0].item()
                prob_continue = probabilities[0][1].item()

            else:
                raise ValueError(
                    f"Expected binary classification (2 classes), got {num_classes} classes"
                )

        return prob_finished


def main():
    print("starting")

    test_cases = [
        "How can I help you today?",
        "Well, I was looking into the available rental cars on your website. And, umm...",
        "Hmm, maybe ",
        "Thank you very much for your help today.",
        "I think that covers everything I needed.",
        "So basically what I'm trying to",
        "Okay, goodbye!",
        "Let me think about this for a",
        "Perfect, that's exactly what I wanted.",
    ]

    evaluator = SemanticEvaluator()

    for text in test_cases:
        prob = evaluator.evaluate(text)
        print(f"{prob:.3f} | {text}")


if __name__ == "__main__":
    main()
