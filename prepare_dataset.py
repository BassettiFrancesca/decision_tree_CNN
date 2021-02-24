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
        for l in range(len(groups)):
            for m in range(len(groups[l])):
                if label == groups[l][m]:
                    indices.append(i)

    data_set = torch.utils.data.Subset(train_set, indices)

    new_dataset = selected_dataset.SelectedDataset(data_set, groups)

    even_i = []
    odd_i = []

    for i in range(len(new_dataset)):
        if i % 2:
            odd_i.append(i)
        else:
            even_i.append(i)

    left_dataset = torch.utils.data.Subset(new_dataset, odd_i)

    right_dataset = torch.utils.data.Subset(new_dataset, even_i)

    return new_dataset, left_dataset, right_dataset
