import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from preprocessing import encode


class ReviewDataset(Dataset):
    def __init__(self, frame, vocabulary, max_length):
        self.sequences = [torch.tensor(encode(text, vocabulary, max_length), dtype=torch.long)
                          for text in frame.text]
        self.labels = torch.tensor(frame.label.eq("positive").to_numpy().astype(np.float32))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate_reviews(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(labels)


class SequenceClassifier(nn.Module):
    def __init__(self, vocabulary_size, architecture, embedding_dim=100, hidden_size=64, dropout=.2):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        recurrent_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[architecture]
        self.recurrent = recurrent_class(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, tokens, lengths):
        output, _ = self.recurrent(self.embedding(tokens))
        # Mô hình một chiều: lấy timestep cuối thực sự, không lấy hidden sau PAD.
        last = output[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.classifier(self.dropout(last)).squeeze(-1)
