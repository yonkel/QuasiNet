import torch
from torcheval.metrics import BinaryAccuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score

from utils.stuff import make_Kaggle_file


def train(epochs, train_set, test_set, model, optimizer, criterion, batch_size, test_function, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps" for macOS

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    training_loss_list = []
    training_accuracy_list = []
    training_f1_score_list = []

    test_accuracy_list = []
    test_f1_score_list = []

    for epoch in range(epochs):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        if scheduler:
            scheduler.step()

        training_loss_list.append(running_loss/len(train_set))
        print(f'Epoch {epoch + 1}, loss = {running_loss / len(train_set)}')

        if epoch % 10 == 0 or epoch == epochs - 1 or True: # Set by preference
            # Testing
            model.eval()
            is_last_epoch = epoch == epochs - 1
            with torch.no_grad():
                test_accuracy, test_f1 = test_function(model, test_loader, device, is_last_epoch, save=is_last_epoch)
                test_accuracy_list.append(test_accuracy)
                test_f1_score_list.append(test_f1)

                train_accuracy, train_f1 = test_function(model, train_loader, device, is_last_epoch)
                training_accuracy_list.append(train_accuracy)
                training_f1_score_list.append(train_f1)
            model.train()
            # print(f'Test Accuracy: at epoch {epoch}: {(100 * test_accuracy)}' )
            # print(f'Train Accuracy: at epoch {epoch}: {(100 * train_accuracy)}' )

    train_eval_metrics = {
        "ACC": training_accuracy_list,
        "F1": training_f1_score_list
    }

    test_eval_metrics = {
        "ACC": test_accuracy_list,
        "F1": test_f1_score_list
    }

    return model, training_loss_list, train_eval_metrics, test_eval_metrics



def binary_accuracy_function(test_model, data_loader, device, verbose=False, save=False):
    threshold = 0

    metric = BinaryAccuracy(threshold=threshold)

    labels_all = []
    outputs_all = []

    for data in data_loader:
        inputs, labels = data

        if threshold == 0:
            labels[labels == -1] = 0

        outputs = test_model(inputs.to(device))
        # print("labels", labels.squeeze())
        # print("outputs", outputs.squeeze())

        metric.update(outputs.squeeze(), labels.squeeze())

        labels = labels.cpu().squeeze()
        outputs = torch.where(outputs.cpu().squeeze() < threshold, 0, 1)

        labels_all += labels.tolist()
        outputs_all += outputs.tolist()

    if verbose:
        print(f"Precision: {precision_score(labels_all, outputs_all)}")
        print(f"Recall: {recall_score(labels_all, outputs_all)}")
        print("Outputs:", outputs_all)
        print("Labels:", labels_all)
        print(confusion_matrix(labels_all, outputs_all))

    if save:
        make_Kaggle_file(outputs_all)

    return metric.compute(), f1_score(labels_all, outputs_all)

def accuracy_function(test_model, data_loader, device):
    total = 0
    correct = 0
    for data in data_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = test_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == '__main__':
    ...

