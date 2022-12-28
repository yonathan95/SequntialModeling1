# Set the random seed manually for reproducibility.
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tcn import TCN
from jsb_datasets import load_jsb
import numpy as np
import matplotlib.pyplot as plt

cuda = True
clip = 0.2

torch.manual_seed(1111)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
input_size = 88
X_train, X_valid, X_test = load_jsb(as_tensor=True)

n_channels = [150] * 4
kernel_size = 5
dropout = 0.25

model = TCN(input_size, input_size, n_channels, kernel_size, dropout)

if cuda:
    model.cuda()

criterion = nn.MSELoss()
lr = 1e-3
optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x.unsqueeze(0)).squeeze(0)
            loss = criterion(y, output)
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = criterion(y, output)
        total_loss += loss.item()
        count += output.size(0)

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % 100 == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
            
    return total_loss / count

def plot_losses(train_lose, test_lose, path=None, title="Losses"):
    """
    This function plots the losses
    :param train_lose: train losses of shape (num_epochs,)
    :param test_lose: test losses of shape (num_epochs,)
    :param path: path and name of the file to save the plot
    :param title: title of the plot
    :return:
    """
    # plot losses
    # change figure size
    plt.figure(figsize=(12, 6.5), dpi=250)
    plt.plot(train_lose, label='train', linewidth=2.5)
    plt.plot(test_lose, label='test', linewidth=2.5)
    plt.yscale('log')
    plt.legend()
    # legend font size
    plt.legend(fontsize=20)
    # add greed
    plt.grid()
    # font size ticks
    plt.tick_params(labelsize=19)
    # font size labels
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("Loss", fontsize=25)
    # give a title to the plot
    plt.title(title)
    # save plot by a name
    if path is not None:
        plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    best_tr_loss = 1e8
    tr_loss_list = []
    train_losses = np.zeros(100)
    test_losses = np.zeros(100)
    model_name = "jsb_tcn_best_model.pt"
    for ep in range(0, 100):
        tr_loss = train(ep)
        tloss = evaluate(X_test, name='Test')
        train_losses[ep] = tr_loss
        test_losses[ep] = tloss
        if tr_loss < best_tr_loss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_tr_loss = tr_loss
        if ep > 10 and tr_loss > max(tr_loss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        tr_loss_list.append(tr_loss)

    plot_losses(train_losses, test_losses, "jsb_tcn_losses.pdf", "TCN on JSB_Chorales")
    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test)