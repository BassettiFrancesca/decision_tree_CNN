import torch
import torch.optim as optim
import torch.nn as nn
import CNN


class Node:
    def __init__(self, data_set, left_child, right_child, PATH):
        self.data_set = data_set
        self.left_child = left_child
        self.right_child = right_child
        self.PATH = PATH

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 2

        net = CNN.Net(2).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        left_net = CNN.Net(2).to(device)
        left_net.load_state_dict(torch.load(self.left_child))

        right_net = CNN.Net(2).to(device)
        right_net.load_state_dict(torch.load(self.right_child))

        for epoch in range(num_epochs):

            indices = []

            train_loader = torch.utils.data.DataLoader(self.data_set, shuffle=True, num_workers=2)

            for i, (image, label) in enumerate(train_loader):
                image = image.to(device)
                label = label.to(device)
                l_r_label = label.to(device)

                left_output = left_net(image)
                _, left_predicted = torch.max(left_output.data, 1)

                right_output = right_net(image)
                _, right_predicted = torch.max(right_output.data, 1)

                if (label[0] == left_predicted[0] and label[0] != right_predicted[0]) or (label[0] != left_predicted[0]
                                                                                          and label[0] == right_predicted[0]):

                    if label[0] == left_predicted[0] and label[0] != right_predicted[0]:
                        l_r_label[0] = 0  # left
                        indices.append(i)

                    if label[0] != left_predicted[0] and label[0] == right_predicted[0]:
                        l_r_label[0] = 1  # right
                        indices.append(i)

                    optimizer.zero_grad()
                    output = net(image)
                    loss = criterion(output, l_r_label)
                    loss.backward()
                    optimizer.step()

            self.data_set = torch.utils.data.Subset(self.data_set, indices)

        print('Finished Training')

        torch.save(net.state_dict(), self.PATH)
