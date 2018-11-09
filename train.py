import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score


def train(model, train_iter, valid_iter, num_epoch=10, lr=1e-3, cuda=True, log_interval=5, valid_interval=5):
    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_auc = []
    valid_losses = []
    valid_auc = []
    model.train()
    steps = 0

    for i in range(1, num_epoch + 1):
        batch_num = 0
        for batch in train_iter:
            sentences, labels = batch.text, batch.label
            batch_size = sentences.size(0)
            batch_num += 1
            if cuda:
                sentences = sentences.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            predictions = model(sentences)
            loss = F.binary_cross_entropy(predictions, labels)

            loss.backward()
            optimizer.step()

            steps += 1

            if steps % log_interval == 0:
                train_loss = loss.data[0]
                train_losses.append(train_loss)

                auc = roc_auc_score(labels.data[0], predictions.data[0])
                train_auc.append(auc)

                print('TRAINING - Epoch {} - Batch {}/{} : loss: {}, auc: {}'.format(
                    i,
                    batch_num,
                    batch_size,
                    train_loss,
                    auc
                ))

            if steps % valid_interval == 0:
                avg_loss, avg_auc = valid(model, valid_iter, cuda)
                valid_losses.append(avg_loss)
                valid_auc.append(avg_auc)

                print('VALIDATION - Epoch {} - Batch {}/{} : loss: {}, auc: {}'.format(
                    i,
                    batch_num,
                    batch_size,
                    avg_loss,
                    avg_auc
                ))

    return train_losses, train_auc, valid_losses, valid_auc


def valid(model, valid_iter, cuda):
    model.eval()

    avg_loss, avg_auc = 0, 0

    for batch in valid_iter:
        sentences, labels = batch.text, batch.label
        if cuda:
            sentences = sentences.cuda()
            labels = labels.cuda()

        predictions = model(sentences)
        loss = F.binary_cross_entropy(predictions, labels)

        avg_loss += loss.data[0]
        avg_auc += roc_auc_score(labels.data[0], predictions.data[0])

    size = len(valid_iter.dataset)
    avg_loss /= size
    avg_auc /= size

    return avg_loss, avg_auc
