import torch
import torchvision
import torchvision.transforms as transforms
import selected_dataset


def prepare_dataset(groups):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=2)

    indices = []

    for i, (image, label) in enumerate(train_loader):
        for j in range(len(groups)):
            for k in range(len(groups[j])):
                if label == groups[j][k]:
                    indices.append(i)

    data_set = torch.utils.data.Subset(train_set, indices)

    new_dataset = selected_dataset.SelectedDataset(data_set, groups)

    print(f'Length new_dataset: {len(new_dataset)}')

    node_i = []
    left_i = []
    right_i = []

    for i in range(0, len(new_dataset), 3):
        node_i.append(i)
    for j in range(1, len(new_dataset), 3):
        left_i.append(j)
    for k in range(2, len(new_dataset), 3):
        right_i.append(k)

    node_dataset = torch.utils.data.Subset(new_dataset, node_i)

    print(f'Length node_dataset: {len(node_dataset)}')

    left_dataset = torch.utils.data.Subset(new_dataset, left_i)

    print(f'Length left_dataset: {len(left_dataset)}')

    right_dataset = torch.utils.data.Subset(new_dataset, right_i)

    print(f'Length right_dataset: {len(right_dataset)}')

    return node_dataset, left_dataset, right_dataset
