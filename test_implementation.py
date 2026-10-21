import unittest

import torch

from models import SequenceClassifier, collate_reviews
from preprocessing import tokenize, build_vocabulary, encode
from rnn_cell import SimpleRNNCell, check_framework_parity


class ImplementationTests(unittest.TestCase):
    def test_cell_matches_framework_forward_and_backward(self):
        report = check_framework_parity()
        self.assertLessEqual(report["output_max_abs_error"], 1e-12)
        self.assertLessEqual(report["input_gradient_max_abs_error"], 1e-12)
        self.assertLessEqual(report["hidden_gradient_max_abs_error"], 1e-12)

    def test_cell_multiple_timesteps_and_weight_gradients(self):
        torch.manual_seed(42)
        custom = SimpleRNNCell(3, 5).double()
        reference = torch.nn.RNNCell(3, 5).double()
        with torch.no_grad():
            reference.weight_ih.copy_(custom.weight_ih)
            reference.weight_hh.copy_(custom.weight_hh)
            reference.bias_ih.copy_(custom.bias)
            reference.bias_hh.zero_()
        x = torch.randn(7, 2, 3, dtype=torch.float64)
        actual = expected = torch.zeros(2, 5, dtype=torch.float64)
        for item in x:
            actual, expected = custom(item, actual), reference(item, expected)
            torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(custom.weight_ih.grad, reference.weight_ih.grad)
        torch.testing.assert_close(custom.weight_hh.grad, reference.weight_hh.grad)
        torch.testing.assert_close(custom.bias.grad, reference.bias_ih.grad)

    def test_padding_does_not_change_prediction(self):
        torch.manual_seed(42)
        for architecture in ["RNN", "LSTM", "GRU"]:
            model = SequenceClassifier(20, architecture, embedding_dim=8, hidden_size=5).eval()
            tokens = torch.tensor([[2, 3, 4], [3, 5, 0]])
            lengths = torch.tensor([3, 2])
            padded = torch.nn.functional.pad(tokens, (0, 7))
            with torch.inference_mode():
                torch.testing.assert_close(model(tokens, lengths), model(padded, lengths), atol=1e-6, rtol=1e-6)

    def test_tokenizer_preserves_negation_and_order(self):
        self.assertEqual(tokenize("I did NOT like it.<br /> It isn't good!"),
                         ["i", "did", "not", "like", "it", "it", "isn't", "good"])

    def test_vocabulary_uses_only_provided_training_text(self):
        vocabulary = build_vocabulary(["good movie", "good acting"], max_size=10, min_frequency=1)
        self.assertEqual(vocabulary["<PAD>"], 0)
        self.assertEqual(vocabulary["<UNK>"], 1)
        self.assertEqual(encode("unseen movie good", vocabulary, 2), [1, vocabulary["movie"]])
        self.assertNotIn("unseen", vocabulary)
        with self.assertRaises(ValueError):
            encode("<br /> 123 !!!", vocabulary, 100)

    def test_collation_retains_true_lengths(self):
        tokens, lengths, labels = collate_reviews([
            (torch.tensor([2, 3]), torch.tensor(1.)), (torch.tensor([4]), torch.tensor(0.))])
        self.assertEqual(tokens.tolist(), [[2, 3], [4, 0]])
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertEqual(labels.tolist(), [1., 0.])


if __name__ == "__main__":
    unittest.main()
