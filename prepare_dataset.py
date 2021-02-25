import torch
import torchvision
import torchvision.transforms as transforms
import selected_dataset


def prepare_dataset(groups):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=2)

    train_indices = []

    test_indices = []

    for i, (image, label) in enumerate(train_loader):
        for j in range(len(groups)):
            for k in range(len(groups[j])):
                if label[0] == groups[j][k]:
                    train_indices.append(i)

    for i, (image, label) in enumerate(test_loader):
        for j in range(len(groups)):
            for k in range(len(groups[j])):
                if label[0] == groups[j][k]:
                    test_indices.append(i)

    train_data_set = torch.utils.data.Subset(train_set, train_indices)

    train_new_dataset = selected_dataset.SelectedDataset(train_data_set, groups)

    print(f'Length train_new_dataset: {len(train_new_dataset)}')

    test_data_set = torch.utils.data.Subset(test_set, test_indices)

    test_new_dataset = selected_dataset.SelectedDataset(test_data_set, groups)

    print(f'Length test_new_dataset: {len(test_new_dataset)}')

    node_i = []
    left_i = []
    right_i = []

    for i in range(0, len(train_new_dataset), 3):
        node_i.append(i)
    for j in range(1, len(train_new_dataset), 3):
        left_i.append(j)
    for k in range(2, len(train_new_dataset), 3):
        right_i.append(k)

    node_dataset = torch.utils.data.Subset(train_new_dataset, node_i)

    print(f'Length node_dataset: {len(node_dataset)}')

    left_dataset = torch.utils.data.Subset(train_new_dataset, left_i)

    print(f'Length left_dataset: {len(left_dataset)}')

    right_dataset = torch.utils.data.Subset(train_new_dataset, right_i)

    print(f'Length right_dataset: {len(right_dataset)}')

    return node_dataset, left_dataset, right_dataset, test_new_dataset
