import torch
import torchvision
import torchvision.transforms as transforms
import group_dataset


def prepare_dataset(groups):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=2)

    indices = []

    for i, (image, label) in enumerate(train_loader):
        for l in range(len(groups)):
            if label == groups[l]:
                indices.append(i)

    dataset = torch.utils.data.Subset(train_set, indices)

    new_dataset = group_dataset.GroupDataset(dataset, groups)

    return new_dataset
