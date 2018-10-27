from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class TextDataset(Dataset):

    def __init__(self, labels, sequences):
        self.labels = list(map(lambda label: 1 if label == 'C' else 0, labels))
        self.sequences = sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor([self.labels[idx]], dtype=torch.float)
