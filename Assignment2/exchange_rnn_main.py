# Set the random seed manually for reproducibility.
import torch
import torch.nn as nn
import torch.optim as optim
from exchange_rnn import RNN
from exchange_dataset import get_exhange_dataloader
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss
import matplotlib.pyplot as plt

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    plt.legend()
    # legend font size
    plt.legend(fontsize=20)
    # add greed
    plt.grid()
    # font size ticks
    plt.tick_params(labelsize=19)
    # font size labels
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("MSE Loss", fontsize=25)
    # give a title to the plot
    plt.title(title)
    # save plot by a name
    if path is not None:
        plt.savefig(path)
    plt.show()

def evaluate(model, data_set, loss_fn, batch_size):
    """
    Evaluate the model on the test set
    :param model: model to evaluate
    :param data_set: data loader for the test set
    :param loss_fn: loss function should be the MSELoss
    :return: average loss on the test set
    """
    model.eval()
    total_loss = 0
    for i, (x, y) in enumerate(tqdm(data_set, leave=False)):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred, _ = model(x)  # y_pred is of shape (batch_size, seq_len, 2)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
    return total_loss / (len(data_set) / batch_size)


def train(model, train_set, loss_fn, opt, batch_size):
    """
    Train the model for 1 epoch
    :param model: model to train
    :param train_set: train set
    :param loss_fn: loss function should be the MSELoss
    :param opt: optimizer
    :return: total loss on the train set
    """
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(tqdm(train_set)):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_hat, _ = model(x)
        loss = loss_fn(y, y_hat)
        loss.backward()
        opt.step()
        total_loss += loss.detach().item()
    return total_loss / (len(train_set) / batch_size)

if __name__ == '__main__':
    v_mag = 1
    delta_t = 0.3

    epochs = 100
    batch_size = 32
    model = RNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    loss_fn = MSELoss().to(device)
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_score = np.inf
    for epoch in tqdm(range(epochs), leave=False):
        # train model
        train_losses[epoch] = train(model, get_exhange_dataloader(), loss_fn, opt, batch_size)
        # test model
        score = evaluate(model, get_exhange_dataloader(flag='test'), loss_fn, batch_size)
        test_losses[epoch] = score
        tqdm.write(f"Epoch {epoch + 1}/{epochs} | Test loss: {score}")
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), "best_model.pt")

    plot_losses(train_losses, test_losses, "losses.pdf")
