import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from modules.quasi import Quasi
from utils.transfer_learning_trainer import train, binary_accuracy_function
from utils.utils import get_bin_cifar_dataset
from Nets import *




if __name__ == "__main__":

    # Fine-tuning
    # 2nd training - Train full net on BinaryCifar with QuasiNet in FC parts

    model = Net_test()

    state_dict = torch.load("model_weights/covn_cifar10.pth")
    model.load_state_dict(state_dict)
    print("loaded model \n", model)

    # Replace last layer with our QuasiNet
    model.fc3 = Quasi(64, 1)
    print("added Quasi layer \n", model)

    model.train()

    # Hyperparameters
    batch_size = 4
    epochs = 1
    learning_rate = 0.005

    transform = transforms.Compose([
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset, test_dataset = get_bin_cifar_dataset(transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Freeze the weights if needed
    # model.requires_grad = False
    # model.fc1.requires_grad = True
    # model.fc2.requires_grad = True
    # model.fc3....

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    print("Starting training...")
    start = time.time()
    model = train(epochs, train_loader, test_loader, model, optimizer, criterion, binary_accuracy_function, squeeze_output=True)
    end = time.time()
    print("Training time: ", end - start)
