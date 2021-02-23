import torch
import torch.optim as optim
import torch.nn as nn
import CNN


class Node:
    def __init__(self, dataset, left_child, right_child, PATH):
        self.dataset = dataset
        self.left_child = left_child
        self.right_child = right_child
        self.PATH = PATH

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = 1
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 2

        net = CNN.Net(3).to(device)

        train_loader = torch.utils.data.DataLoader(self.dataset, shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        left_net = CNN.Net(2).to(device)
        left_net.load_state_dict(torch.load(self.left_child))

        right_net = CNN.Net(2).to(device)
        right_net.load_state_dict(torch.load(self.right_child))

        for epoch in range(num_epochs):

            for (inputs, labels) in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                l_r_labels = labels.to(device)

                left_outputs = left_net(inputs)
                _, left_predicted = torch.max(left_outputs.data, 1)

                right_outputs = right_net(inputs)
                _, right_predicted = torch.max(right_outputs.data, 1)

                for l in range(batch_size):
                    label = labels[l]
                    left_pred = left_predicted[l]
                    right_pred = right_predicted[l]
                    l_r_labels[l] = 2  # don't care
                    if label == left_pred and label != right_pred:
                        l_r_labels[l] = 0  # left
                    if label != left_pred and label == right_pred:
                        l_r_labels[l] = 1  # right

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, l_r_labels)
                loss.backward()
                optimizer.step()

        print('Finished Training')

        torch.save(net.state_dict(), self.PATH)
