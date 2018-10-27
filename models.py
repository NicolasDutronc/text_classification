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
