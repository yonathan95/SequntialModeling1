import torch
import matplotlib.pyplot as plt
import numpy as np
from model import MyModel


def predict_trajectory(path, step_num=20, num_points=10000):
    """
    This function predicts the trajectory of the harmonic oscillator
    :param path: path to the model
    :param step_num: number of steps to predict
    :param num_points: number of points to predict
    :return: predicted trajectory of shape (num_points, step_num, 2)
    """
    # load model state from model.pt
    state = torch.load(path)
    model = MyModel()
    model.load_state_dict(state)
    model.eval()
    # predict 10000 samples of length 20
    x_init = torch.rand(num_points, 1, 2)
    return model.inference(x_init, steps=step_num)


def plot_phase_portrait_and_trajectory(x, v, path=None, nth=2, title="Phase portrait and trajectory"):
    """
    This function plots the phase portrait and the trajectory of the harmonic oscillator
    :param x: x coordinates of the trajectory
    :param v: velocity coordinates of the trajectory
    :param path: path and name of the file to save the plot
    :param nth: plot every nth vector
    :param title: title of the plot
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=250)
    ax.plot(x.T, v.T, alpha=0.5)
    vector_x, vector_v = x[:, 1:] - x[:, :-1], v[:, 1:] - v[:, :-1]
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("velocity")
    # set a fine grid with bold lines
    ax.set_xticks(np.arange(-1.2, 1.201, 0.2))
    ax.set_yticks(np.arange(-1.2, 1.201, 0.2))
    ax.grid(linestyle='-', linewidth='1.5')
    # set lims
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    # font size ticks
    ax.tick_params(labelsize=22)
    # font size labels
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    # draw every nth vector
    ax.quiver(x[::5, :-1:nth], v[::5, :-1:nth], vector_x[::5, ::nth], vector_v[::5, ::nth],
              scale=1.5, scale_units="xy", width=0.004)

    # plot axes
    ax.axhline(y=0, color='k', linewidth=1.8)
    ax.axvline(x=0, color='k', linewidth=1.8)
    # give a title to the plot
    plt.title(title, fontsize=30, y=-0.12)

    if path is not None:
        plt.savefig(path)
    plt.show()


def predict_and_plot_phase_portrait_and_trajectory(model_path, num_points=100, step_num=20, nth=2, save_path=None, title='Phase portrait and trajectory'):
    """
    This function predicts the trajectory of the harmonic oscillator and plots the phase portrait and the trajectory
    :param model_path: path to the model
    :param num_points: number of points to predict
    :param step_num: number of steps to predict
    :param nth: plot every nth vector
    :param save_path: path and name of the file to save the plot
    :return:
    """
    predicted = predict_trajectory(model_path, step_num, num_points)
    plot_phase_portrait_and_trajectory(predicted[:, :, 0], predicted[:, :, 1], path=save_path, nth=nth, title=title)


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


if __name__ == '__main__':
    # plot phase portrait and trajectory
    predict_and_plot_phase_portrait_and_trajectory("best_model.pt", num_points=200, step_num=20, nth=2)
