import time

import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils.Nets import *
from utils.transfer_learning_trainer import train, accuracy_function

transform = transforms.Compose([
    transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == "__main__":

    # Feature extraction
    # 1st training - Full net on Cifar10 without QuasiNet

    batch_size = 4
    epochs = 15
    learning_rate = 0.005
    # momentum = 0.8

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = TestNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(model)

    start = time.time()
    model = train(epochs, train_loader, test_loader, model, optimizer, criterion, accuracy_function)
    end = time.time()

    print("Training time: ", end - start)


    torch.save(model.state_dict(), 'model_weights/covn_cifar10.pth')
