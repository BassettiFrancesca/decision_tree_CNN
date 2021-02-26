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

    index_groups = [[] for i in range(len(groups))]

    for i, (image, label) in enumerate(train_loader):
        for j in range(len(groups)):
            if label[0] in groups[j]:
                index_groups[j].append(i)

    node_i = []
    leaf_i = []

    for i in range(len(index_groups)):
        for j in range(0, len(index_groups[i]), 2):
            node_i.append(index_groups[i][j])
        for k in range(1, len(index_groups[i]), 2):
            leaf_i.append(index_groups[i][k])

    node_dataset = torch.utils.data.Subset(train_set, node_i)

    train_node_dataset = selected_dataset.SelectedDataset(node_dataset, groups)

    print(f'Size node_dataset: {len(train_node_dataset)}')

    leaf_dataset = torch.utils.data.Subset(train_set, leaf_i)

    train_leaf_dataset = selected_dataset.SelectedDataset(leaf_dataset, groups)

    print(f'Size leaf_dataset: {len(train_leaf_dataset)}')

    test_indices = []

    for i, (image, label) in enumerate(test_loader):
        for j in range(len(groups)):
            if label[0] in groups[j]:
                test_indices.append(i)

    test_data_set = torch.utils.data.Subset(test_set, test_indices)

    test_new_dataset = selected_dataset.SelectedDataset(test_data_set, groups)

    print(f'Size test_new_dataset: {len(test_new_dataset)}')

    return train_node_dataset, train_leaf_dataset, test_new_dataset
