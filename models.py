import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):

    def __init__(self, vocab_size, embeddings_size):
        super(SimpleModel, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embeddings_size)
        self.linear = nn.Linear(embeddings_size, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # print('x shape:', x.shape)
        # print('x:', x)
        embeds = self.embeddings(x).squeeze()
        # print('embeds shape:', embeds.shape)
        # print('embeds:', embeds)
        hidden = F.relu(self.linear(embeds))
        # print('hidden shape:', hidden.shape)
        # print('hidden:', hidden)
        pred = torch.sigmoid(self.out(hidden))
        # print('pred shape:', pred.shape)
        # print('pred:', pred)
        return pred


class CNNModel(nn.Module):

    def __init__(self, vocab_size, embeddings_size, kernel_sizes, kernel_num, dropout):
        super(CNNModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embeddings_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embeddings_size)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, 1)

    def forward(self, x):
        # print()
        # print('initial shape:', x.shape)
        x = self.embeddings(x)  # (N, max_len, embeddings_size)
        # print('shape after embeddings:', x.shape)
        x = x.unsqueeze(1)  # (N, 1, max_len, embeddings_size)
        # print(self.convs[0](x).shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, kernel_num, max_len)] * len(kernel_sizes)
        # print('length of x after convs:', len(x))
        # print('shape of an element of x after convs:', x[0].shape)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, kernel_num)] * len(kernel_sizes)
        # print('shape of an element of x after pooling:', x[0].shape)
        x = torch.cat(x, 1)  # (N, len(kernel_sizes) * kernel_num)
        # print('shape of x after cat:', x.shape)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))  # (N, 1)
