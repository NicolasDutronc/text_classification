import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):

    def __init__(self, vocab_size, embeddings_size):
        super(SimpleModel, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embeddings_size)
        self.linear = nn.Linear(embeddings_size, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        embeds = self.embeddings(x.unsqueeze(0)).squeeze()
        hidden = F.relu(self.linear(embeds))
        return self.out(hidden)
