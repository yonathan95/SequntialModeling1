import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import random_split
from tqdm import tqdm

from datasets import HarmonicOscillator
from model import MyModel
from viz import plot_losses, predict_and_plot_phase_portrait_and_trajectory
from viz import plot_phase_portrait_and_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


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
    for i in tqdm(range(0, len(data_set), batch_size), leave=False):
        x, y = data_set[i:i + batch_size]
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
    indices = np.arange(len(train_set))
    np.random.shuffle(indices)  # shuffle the indices
    for i in tqdm(range(0, len(train_set), batch_size), leave=False):
        x, y = train_set[list(indices[i:i + batch_size])]
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
    data = HarmonicOscillator(delta_t, v_magnitude=v_mag)
    trainset, testset = random_split(data, [int(0.8 * len(data)), int(0.2 * len(data))])
    xs, _ = trainset[:]
    # plot the phase portrait and trajectory of the train set
    plot_phase_portrait_and_trajectory(xs[:, :, 0], xs[:, :, 1], path='train_set_phase_portrait.pdf',
                                       title=f'')

    xs, _ = testset[:]
    plot_phase_portrait_and_trajectory(xs[:, :, 0], xs[:, :, 1], path='test_set_phase_portrait.pdf',
                                       title='')

    epochs = 100
    batch_size = 32
    model = MyModel().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    loss_fn = MSELoss().to(device)
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_score = np.inf
    for epoch in tqdm(range(epochs), leave=False):
        # train model
        train_losses[epoch] = train(model, trainset, loss_fn, opt, batch_size)
        # test model
        score = evaluate(model, testset, loss_fn, batch_size)
        test_losses[epoch] = score
        tqdm.write(f"Epoch {epoch + 1}/{epochs} | Test loss: {score}")
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), "best_model.pt")

    plot_losses(train_losses, test_losses, "losses.pdf")

    # plot the phase portrait and trajectory of randomly sampled data for [0, 1]x[0, 1]
    predict_and_plot_phase_portrait_and_trajectory('best_model.pt', num_points=400, save_path='phase_portrait_an_400_points.pdf', title='')
    predict_and_plot_phase_portrait_and_trajectory('best_model.pt', num_points=100, save_path='phase_portrait_an_100_points.pdf', title='')
