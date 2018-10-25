from torch.utils.data import Dataset
import pandas as pd
import torch


class TextDataset(Dataset):

    def __init__(self, labels, sequences):
        self.df = pd.DataFrame()
        self.df['labels'] = pd.Series(labels).apply(lambda s: 1 if s == 'C' else 0)
        self.df['sequences'] = pd.Series(sequences)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        # print(item)
        return torch.LongTensor(item.sequences), torch.LongTensor([item.labels])
