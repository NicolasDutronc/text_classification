import torch.nn.functional as F
import torch.optim as optim
import torch

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

            # print(predictions.size())
            # print(labels.unsqueeze(1).size())
            # print(labels)
            loss = F.binary_cross_entropy(predictions.squeeze(), labels.float())

            loss.backward()
            optimizer.step()

            steps += 1

            with torch.no_grad():

                if steps % log_interval == 0:
                    train_loss = loss.item()
                    train_losses.append(train_loss)

                    # print(labels.size())
                    # print(predictions.size())

                    auc = 0
                    if cuda:
                        auc = roc_auc_score(labels.cpu().detach().numpy(), predictions.squeeze().cpu().detach().numpy())
                    else:
                        auc = roc_auc_score(labels.detach().numpy(), predictions.squeeze().detach().numpy())
                    train_auc.append(auc)

                    print('TRAINING - Epoch {}/{} - Batch {}/{} : loss: {}, auc: {}'.format(
                        i,
                        num_epoch,
                        batch_num,
                        len(train_iter),
                        train_loss,
                        auc
                    ))

                if steps % valid_interval == 0:
                    avg_loss, avg_auc = valid(model, valid_iter, cuda)
                    valid_losses.append(avg_loss)
                    valid_auc.append(avg_auc)

                    print('VALIDATION - Epoch {}/{} - Batch {}/{} : loss: {}, auc: {}'.format(
                        i,
                        num_epoch,
                        batch_num,
                        len(train_iter),
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

        # print(labels)

        predictions = model(sentences)
        loss = F.binary_cross_entropy(predictions.squeeze(), labels.float())

        avg_loss += loss.item()
        if cuda:
            avg_auc += roc_auc_score(labels.cpu().detach().numpy(), predictions.squeeze().cpu().detach().numpy())
        else:
            avg_auc += roc_auc_score(labels.detach().numpy(), predictions.squeeze().detach().numpy())

    size = len(valid_iter)
    avg_loss /= size
    avg_auc /= size

    return avg_loss, avg_auc
