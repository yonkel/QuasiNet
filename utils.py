import torch
import itertools
import numpy as np

from torch.utils.data import Dataset

def twospirals_raw(n_points, noise=0.5):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))

def spirals(points, test_size=0.2):
    x, y = twospirals_raw(points)
    m = np.max(x)
    x = x / m
    y = np.where(y == 0, -1, y)
    y = np.reshape(y, (len(y), 1))
    return x, y


class make_into_set(Dataset):
    def __init__(self, X, y):
        # X, y = spirals(150)
        #TODO make this better
        if type(X) != torch.Tensor:
            X = torch.tensor(X)
        if type(y) != torch.Tensor:
            y = torch.tensor(y)

        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]), torch.Tensor(self.y[idx])


def get_parity_dataset(degree : int, remap=False):
    X = [x for x in itertools.product([0,1], repeat=degree)]

    pt_X = torch.tensor(X)
    pt_labels = torch.sum(pt_X, axis=1) % 2

    if remap:
        pt_X = torch.where(pt_X == 0, -1, 1)
        pt_labels = torch.where(pt_labels == 0, -1, 1)

    pt_X = pt_X.to(torch.float)
    pt_labels = pt_labels.to(torch.float)

    return make_into_set(pt_X, pt_labels)

if __name__ == '__main__':
    set = get_parity_dataset(3, remap=True)

    for item in set:
        print(item)