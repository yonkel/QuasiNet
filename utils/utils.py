import torch
import itertools
from torchvision import datasets, transforms
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


# def twospirals_raw(n_points, noise=0.5):
#     n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
#     d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
#     d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
#     return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
#             np.hstack((np.zeros(n_points), np.ones(n_points))))
#
# def spirals(points, test_size=0.2):
#     x, y = twospirals_raw(points)
#     m = np.max(x)
#     x = x / m
#     y = np.where(y == 0, -1, y)
#     y = np.reshape(y, (len(y), 1))
#     return x, y


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
    pt_labels = pt_labels.unsqueeze(1).to(torch.float)

    return make_into_set(pt_X, pt_labels)

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            label = self.target_transform(label)

        return data, label

def get_dataset_from_HF(name: str, transform: transforms = None, target_transform: transforms = None):
    hf_data = load_dataset(name)

    hf_train = hf_data["train"]

    if "test" in hf_data:
        hf_test = hf_data["test"]
    else:
        hf_test = hf_data["train"]

    train_set = HFDataset(hf_train['img'], hf_train['label'], transform=transform,
                          target_transform=target_transform)
    test_set = HFDataset(hf_test['img'], hf_test['label'], transform=transform,
                         target_transform=target_transform)

    return train_set, test_set


def get_bin_cifar_dataset(transform):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes_bin = (-1, -1, 1, 1, 1, 1, 1, 1, -1, -1)

    target_transform = transforms.Lambda(
        lambda y: torch.tensor(-1).to(torch.float) if y in [0, 1, 8, 9] else torch.tensor(1).to(torch.float)
    )

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform, target_transform=target_transform)

    # train_set, test_set = get_dataset_from_HF("uoft-cs/cifar10", transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset


if __name__ == '__main__':
    # set = get_parity_dataset(3, remap=True)
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    train_set, test_set = get_bin_cifar_dataset(transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

    for input, label in train_loader:
        print(input, label)
