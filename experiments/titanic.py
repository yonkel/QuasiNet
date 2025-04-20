import time

import torch.optim as optim
import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
import matplotlib.pyplot as plt

from utils.Nets import *
from utils.Nets import TitanicQTNet, TitanicTQNet
from utils.data import get_titanic_dataset
from utils.stuff import export_to_latex, make_plt_graph
from utils.transfer_learning_trainer import train, binary_accuracy_function

if __name__ == "__main__":
    # Setup
    model = TitanicQTQNet()

    print(model)
    name = "QTQ"

    epochs =  50
    batch_size = 32

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01 , momentum=0.5)

    train_set, test_set = get_titanic_dataset(remap=True)

    # seed = 42
    # generator = torch.Generator().manual_seed(seed)
    # train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1], generator)

    # Training
    start = time.time()
    model, train_loss, train_metrics, test_metrics = train(epochs, train_set, test_set, model, optimizer, criterion, batch_size, binary_accuracy_function)
    end = time.time()
    print("Training time: ", end - start)

    torch.save(model.state_dict(), f'../model_weights/titanic_{name}.pth')

    print(f"Train ACC: {train_metrics['ACC'][-1]}, F1: {train_metrics['F1'][-1]}, loss: {train_loss[-1]}")
    print(f"Test ACC: {test_metrics['ACC'][-1]}, F1: {test_metrics['F1'][-1]}")



    # Plot Train loss
    plt.plot(train_loss, label="Train", color="#4169e1")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.box(False)
    plt.savefig(f"../results/titanic/titanic{name}_train_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()


    # Plot Accuracy
    plt.plot(train_metrics['ACC'], label="Train", color="#4169e1")
    plt.plot(test_metrics['ACC'], label="Test", color='firebrick')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.box(False)
    plt.savefig(f"../results/titanic/titanic{name}_accuracy.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plot F1
    plt.plot(train_metrics['F1'], label="Train", color="#4169e1")
    plt.plot(test_metrics['F1'], label="Test", color='firebrick')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.box(False)
    plt.savefig(f"../results/titanic/titanic{name}_f1.pdf", format="pdf", bbox_inches="tight")
    plt.show()