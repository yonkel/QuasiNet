import time

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

from utils.Nets import *
from utils.Nets import TitanicQuasiNet
from utils.data import get_titanic_dataset
from utils.stuff import export_to_latex, make_plt_graph
from utils.transfer_learning_trainer import train, binary_accuracy_function

if __name__ == "__main__":
    # Setup
    model = TitanicQuasiNet()
    print(model)

    epochs = 200
    batch_size = 32

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005) #, momentum=0.9)


    train_set, test_set = get_titanic_dataset(remap=True)

    # Training
    start = time.time()
    model, train_loss, test_loss, train_acc = train(epochs, train_set, test_set, model, optimizer, criterion, batch_size, binary_accuracy_function)
    end = time.time()
    print("Training time: ", end - start)


    # Plot Train loss
    plt.plot(train_loss, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(f"../results/titanic/titanic_train_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plot Test loss
    plt.plot(train_acc, label="Train")
    plt.plot(test_loss, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"../results/titanic/titanic_test_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    export_to_latex("titanic/train_loss.txt", train_loss)
    export_to_latex("titanic/test_loss.txt", test_loss)