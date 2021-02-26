from torch.utils.data import Dataset


class SelectedDataset(Dataset):

    def __init__(self, dataset, groups):
        self._dataset = dataset
        self._groups = groups

    def __getitem__(self, idx):
        (x, y) = self._dataset[idx]
        for (i, group) in enumerate(self._groups):
            if y in group:
                return x, i

    def __len__(self):
        return len(self._dataset)
