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

    def train(self, num_epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        learning_rate = 0.001
        momentum = 0.9

        net = CNN.Net().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        left_net = CNN.Net().to(device)
        left_net.load_state_dict(torch.load(self.left_child))

        right_net = CNN.Net().to(device)
        right_net.load_state_dict(torch.load(self.right_child))

        left = 0
        right = 0
        dcw = 0
        dcr = 0

        for epoch in range(num_epochs):

            indices = []

            train_loader = torch.utils.data.DataLoader(self.data_set, shuffle=False, num_workers=2)

            #print(f'Length dataset: {len(train_loader)}')

            for i, (image, label) in enumerate(train_loader):
                image = image.to(device)
                label = label.to(device)
                l_r_label = label.to(device)

                left_output = left_net(image)
                _, left_predicted = torch.max(left_output.data, 1)

                right_output = right_net(image)
                _, right_predicted = torch.max(right_output.data, 1)

                if right_predicted[0] != left_predicted[0]:

                    if label[0] == left_predicted[0]:
                        l_r_label[0] = 0  # left
                        indices.append(i)
                        if epoch == 0:
                            left += 1
                    else:
                        if label[0] == right_predicted[0]:
                            l_r_label[0] = 1  # right
                            indices.append(i)
                            if epoch == 0:
                                right += 1

                    optimizer.zero_grad()
                    output = net(image)
                    loss = criterion(output, l_r_label)
                    loss.backward()
                    optimizer.step()
                else:
                    if epoch == 0:
                        if label[0] == left_predicted[0] and label[0] == right_predicted[0]:
                            dcr += 1
                        else:
                            if label[0] != left_predicted[0] and label[0] != right_predicted[0]:
                                dcw += 1
            self.data_set = torch.utils.data.Subset(self.data_set, indices)
            #print(f'N° indices: {len(indices)}')
        print(f'N° indices: {len(indices)}')
        print('Finished Training')
        print(f'Right: {right}')
        print(f'Left: {left}')
        print(f'DontCareWrong: {dcw}')
        print(f'DontCareRight: {dcr}')

        torch.save(net.state_dict(), self.PATH)
