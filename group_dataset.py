from torch.utils.data import Dataset


class GroupDataset(Dataset):

    def __init__(self, dataset, groups):
        self._dataset = dataset
        self._groups = groups

    def __getitem__(self, idx):
        (x, y) = self._dataset[idx]
        for i in range(len(self._groups)):
            if y == self._groups[i]:
                return x, i

    def __len__(self):
        return len(self._dataset)
