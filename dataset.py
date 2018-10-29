import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, labels, sequences):
        self.labels = labels
        self.sequences = sequences

    def index_labels(self):
        self.labels = list(map(lambda label: 1 if label == 'C' else 0, self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor([self.labels[idx]], dtype=torch.float)

    def split(self, train_ratio):
        x_train, x_test, y_train, y_test = train_test_split(self.sequences, self.labels, train_size=train_ratio)
        return TextDataset(y_train, x_train), TextDataset(y_test, x_test)
