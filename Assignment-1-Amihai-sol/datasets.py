import torch
from torch.utils.data import Dataset, random_split
import numpy as np
from viz import plot_phase_portrait_and_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class HarmonicOscillator(Dataset):

    def __init__(self, delta_t: float, num_samples=500, steps_num=20, v_magnitude=1.):
        """
        :param delta_t: time step
        :param num_samples: number of samples
        :param steps_num: number of steps in each sample
        :param v_magnitude: initial velocity magnitude
        """
        assert 0 < delta_t < 1, "0 < ∆t << 1"
        self.delta_t = delta_t
        self.steps_num = steps_num
        x_0 = torch.rand(num_samples)  # initial condition for x_0
        v_0 = torch.rand(num_samples) * v_magnitude  # initial condition for v_0
        self.data = torch.zeros(num_samples, steps_num + 1, 2)
        self.data[:, 0, 0] = x_0  # Sample_num, Time, [x, v]. x.shape == 1 , v.shape == 1
        self.data[:, 0, 1] = v_0
        x_1 = self.data[:, 1, 0] = x_0 + v_0 * delta_t - 0.5 * x_0 * delta_t ** 2
        self.data[:, 1, 1] = (x_1 - x_0) / delta_t  # v_1 = (x_1 - x_0) / delta_t
        for i in range(2, steps_num + 1):
            x_t = self.data[:, i - 1, 0]  # xt
            x_t_m1 = self.data[:, i - 2, 0]  # xt-1
            self.data[:, i, 0] = 2 * x_t - x_t_m1 - x_t * delta_t ** 2  # xt+1
            self.data[:, i, 1] = (self.data[:, i, 0] - x_t) / delta_t  # vt+1

    def __getitem__(self, index):
        """
        :param index: index of the sample
        :return: x, y of shape (seq_len, 2)
        """
        return self.data[index, :-1, :], self.data[index, 1:, :]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    delta_t = 0.3
    v_mag = 1
    data = HarmonicOscillator(delta_t, steps_num=20, v_magnitude=v_mag, num_samples=500)
    train, test = random_split(data, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    print(len(train), len(test))
    n = data.steps_num
    to_plot_num = 100
    data, _ = train[:to_plot_num]
    x, y = data[..., 0], data[..., 1]
    plot_phase_portrait_and_trajectory(x, y)





