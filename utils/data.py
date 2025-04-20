import torch
import itertools
from torchvision import datasets, transforms
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import numpy as np
import pandas as pd


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


class ListDataset(Dataset):
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

    return ListDataset(pt_X, pt_labels)


class DataFrameDataset(Dataset):
    def __init__(self, df, label_col):
        feature_cols = [key for key in df.keys() if key != label_col]

        # print(df[feature_cols][:100])
        # print(df[label_col])

        self.features = df[feature_cols].values.astype('float32')
        self.labels = df[label_col].values.astype('float32')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).view(1)

        return features, label

def get_titanic_dataset(remap=False):
    train = pd.read_csv("../data/titanic/train.csv")
    test = pd.read_csv("../data/titanic/test_augmented.csv")


    # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # Keep only relevant features
    keys_to_keep = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
    train = train[keys_to_keep]
    test = test[keys_to_keep]

    # OHE `Sex` and `Embark`
    ohe_sex_train = pd.get_dummies(train["Sex"], dtype="int8")
    ohe_embarked_train = pd.get_dummies(train["Embarked"], dtype="int8")
    ohe_sex_test = pd.get_dummies(test["Sex"], dtype="int8")
    ohe_embarked_test = pd.get_dummies(test["Embarked"], dtype="int8")

    train.drop(["Sex", "Embarked"], axis=1, inplace=True)
    test.drop(["Sex", "Embarked"], axis=1, inplace=True)

    train = pd.concat([train, ohe_sex_train, ohe_embarked_train], axis=1)
    test = pd.concat([test, ohe_sex_test, ohe_embarked_test], axis=1)

    # Fill missing values
    train['Age'] = train['Age'].fillna(train['Age'].median())
    test['Age'] = test['Age'].fillna(test['Age'].median())
    test['Fare'] = test['Fare'].fillna(test['Fare'].median())

    # Create FamilySize feature
    train['FamilySize'] = train['SibSp'] + train['Parch']
    test['FamilySize'] = test['SibSp'] + test['Parch']

    train.drop(["SibSp", "Parch"], axis=1, inplace=True)
    test.drop(["SibSp", "Parch"], axis=1, inplace=True)

    # Divide the `Fare` and `Age` into discrete intervals
    # train['FareBin'] = pd.qcut(train['Fare'], 4, labels=False)
    # test['FareBin'] = pd.qcut(test['Fare'], 4, labels=False)

    train['AgeBin'] = pd.cut(train['Age'], [0, 13, 60, 100], labels=False)
    test['AgeBin'] = pd.cut(test['Age'], [0, 13, 60, 100], labels=False)

    train.drop(columns=["Age",], inplace=True)
    test.drop(columns=["Age"], inplace=True)

    # Remap 0 to -1 for Quasi
    if remap:
        train[["Survived", 'female', 'male', 'C', 'Q', 'S' ]] = train[["Survived", 'female', 'male', 'C', 'Q', 'S']].replace({0: -1})
        test[["Survived", 'female', 'male', 'C', 'Q', 'S']] = test[["Survived", 'female', 'male', 'C', 'Q', 'S']].replace({0: -1})

    return DataFrameDataset(train,"Survived"), DataFrameDataset(test,"Survived")



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
    t, _ = get_titanic_dataset(True)
    print(t[4])