import argparse
import json
from pathlib import Path

import torch

from models import SequenceClassifier, collate_reviews
from preprocessing import ROOT, encode, tokenize


class SentimentPredictor:
    def __init__(self, checkpoint=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if checkpoint is None:
            selection = json.loads((ROOT / "artifacts" / "selection.json").read_text())
            checkpoint = ROOT / selection["application_checkpoint"]
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.vocabulary = saved["vocabulary"]
        self.config = saved["config"]
        self.model = SequenceClassifier(len(self.vocabulary), self.config["architecture"],
                                        self.config["embedding_dim"], self.config["hidden_size"],
                                        self.config["dropout"]).to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text):
        ids = encode(text, self.vocabulary, self.config["max_length"])
        tokens, lengths, _ = collate_reviews([(torch.tensor(ids), torch.tensor(0.))])
        probability = torch.sigmoid(self.model(tokens.to(self.device), lengths.to(self.device))).item()
        words = tokenize(text)
        return {"sentiment": "positive" if probability >= .5 else "negative",
                "positive_probability": probability,
                "predicted_class_probability": probability if probability >= .5 else 1 - probability,
                "input_tokens": len(words), "retained_tokens": len(ids),
                "truncated": len(words) > len(ids),
                "oov_tokens_retained": sum(index == 1 for index in ids)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(SentimentPredictor(args.checkpoint).predict(args.text), ensure_ascii=False, indent=2))
    except ValueError as error:
        parser.error(str(error))
