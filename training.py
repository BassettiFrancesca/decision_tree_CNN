import torch
import torch.optim as optim
import torch.nn as nn
import CNN


def train(train_set, PATH):
    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 2

    net = CNN.Net(2).to(device)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):

        for i, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')

    torch.save(net.state_dict(), PATH)
