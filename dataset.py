import torch
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchtext import data
from spacy.lang.fr.stop_words import STOP_WORDS
import spacy


class TextDataset(Dataset):

    def __init__(self, labels, sequences):
        self.labels = labels
        self.sequences = sequences

    def index_labels(self):
        self.labels = list(map(lambda label: 0 if label == 'C' else 1, self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor([self.labels[idx]], dtype=torch.float)

    def split(self, train_ratio):
        x_train, x_test, y_train, y_test = train_test_split(self.sequences, self.labels, train_size=train_ratio)
        return TextDataset(y_train, x_train), TextDataset(y_test, x_test)


class TorchTextDataset(data.Dataset):

    @staticmethod
    def sort_key(example):
        return len(example.text)

    def __init__(self, sentences, labels):

        # build tokenizer
        spacy_fr = spacy.load('fr')
        tokenizer = lambda s: [token.text for token in spacy_fr.tokenizer(s)]

        # compute max length sentence for padding
        max_len = np.max([len(s) for s in sentences])

        # define fields
        self.text_field = data.Field(
            lower=True,
            tokenize=tokenizer,
            stop_words=STOP_WORDS,
            fix_length=max_len,
            batch_first=True)
        self.label_field = data.Field(sequential=False)
        fields = [('text', self.text_field), ('label', self.label_field)]

        # build the examples
        examples = []
        for i in range(len(sentences)):
            examples.append(data.Example.fromlist([sentences[i], labels[i]], fields))

        # build the dataset
        super(TorchTextDataset, self).__init__(examples, fields)

    def get_iterators(self, split_ratio=0.8, batch_size=64):
        train_data, valid_data = dataset.split(split_ratio=split_ratio)
        self.text_field.build_vocab(train_data, valid_data)
        self.label_field.build_vocab(train_data, valid_data)
        return data.Iterator.splits(
            (train_data, valid_data),
            batch_sizes=(batch_size, len(valid_data))
        )
