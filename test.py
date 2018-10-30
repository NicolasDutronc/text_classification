import pickle as pkl
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from dataset import TextDataset
from models import SimpleModel
from utils.colors import get_cmap

gc.enable()

labels = pkl.load(open('./data/train/labels.pkl', 'rb'))
sentences = pkl.load(open('./data/train/sentences.pkl', 'rb'))
sequences = pkl.load(open('./data/train/sequences.pkl', 'rb'))
vocab = pkl.load(open('./data/dict.pkl', 'rb'))


data = TextDataset(labels, sequences)
data.index_labels()

print(data[1])

model = SimpleModel(len(vocab), 5)
model.cuda()

train, test = data.split(0.7)

# data loaders
train_loader, test_loader = DataLoader(train, shuffle=True), DataLoader(test, shuffle=True)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# loss
loss = nn.CrossEntropyLoss(torch.DoubleTensor([0.86]))

num_epoch = 5
train_losses = []
test_losses = []
cmap = get_cmap(num_epoch)

for i in range(num_epoch):
    print("Epoch:", i)
    epoch_loss = 0
    for sequence, label in train_loader:

        # print(sequence)
        # print(label)

        sequence = sequence.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        prediction = model(sequence)
        output = F.binary_cross_entropy(prediction, label)
        epoch_loss += float(output)
        output.backward()
        optimizer.step()
    print('train_loss:', epoch_loss / len(train_loader.dataset))
    train_losses.append(epoch_loss / len(train_loader.dataset))
    test_loss = 0
    test_labels = []
    test_predictions = []
    gc.collect()

    # test session
    with torch.no_grad():
        for sequence, label in test_loader:

            test_labels += [float(x) for x in label.cpu().numpy().tolist()[0]]

            sequence = sequence.cuda()
            label = label.cuda()

            prediction = model(sequence)

            test_predictions += [float(x) for x in prediction.cpu().numpy()]
            output = F.binary_cross_entropy(prediction, label)
            test_loss += float(output)
        print('test loss:', test_loss / len(test_loader.dataset))
        test_losses.append(test_loss / len(test_loader.dataset))
        # print('test labels:', test_labels)
        # print('test predictions:', test_predictions)

        print('auc:', roc_auc_score(test_labels, test_predictions))

        [fpr, tpr, thr] = roc_curve(test_labels, test_predictions)
        print('sensitivity:', tpr)
        print('specificity:', 1 - fpr)

        plt.plot(fpr, tpr, color=cmap(i), lw=2, label=i)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - specificite', fontsize=14)
        plt.ylabel('Sensibilite', fontsize=14)
        plt.legend()

        gc.collect()

plt.show()

print('train losses:', train_losses)
print('test losses:', test_losses)
