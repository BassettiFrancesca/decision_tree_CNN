from torch.utils.data import Dataset


class SelectedDataset(Dataset):

    def __init__(self, dataset, groups):
        self._dataset = dataset
        self._groups = groups

    def __getitem__(self, idx):
        (x, y) = self._dataset[idx]
        for i in range(len(self._groups)):
            for l in range(len(self._groups[i])):
                if y == self._groups[i][l]:
                    return x, i

    def __len__(self):
        return len(self._dataset)
