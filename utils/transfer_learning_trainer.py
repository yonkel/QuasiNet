import torch

def train(epochs, train_loader, test_loader, model, optimizer, criterion, test_function, squeeze_output=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps" for macOS

    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            if squeeze_output:
                outputs = outputs.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        print(f'Epoch {epoch + 1}, loss = {running_loss / len(train_loader)}')

        if epoch % 5 == 1 or epoch == epochs - 1:
            # Testing
            model.eval()
            with torch.no_grad():
                accuracy = test_function(model, test_loader, device)

            print(f'Test Accuracy: at epoch {epoch}: {(100 * accuracy)}' )
            model.train()

    return model

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


def binary_accuracy_function(test_model, data_loader, device):
    total = 0
    correct = 0
    for data in data_loader:
        inputs, labels = data
        outputs = test_model(inputs).squeeze()
        inputs, labels = inputs.to(device), labels.to(device)

        if outputs.dim() == 0:
            total += 1
            if (outputs > 0 and labels.item() == 1) or (outputs < 0 and labels.item() == -1):
                correct += 1
        else:
            total += outputs.size(0)
            correct += ((outputs > 0) & (labels == 1)).sum().item()
            correct += ((outputs < 0) & (labels == -1)).sum().item()


    return correct / total

if __name__ == '__main__':
    ...