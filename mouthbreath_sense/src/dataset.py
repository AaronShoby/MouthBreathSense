import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MouthBreathDataset(Dataset):
    """Dummy dataset that yields random tensors for prototyping."""

    def __init__(self, length: int = 200, num_classes: int = 2):
        self.length = length
        self.num_classes = num_classes
        print(f"[dataset] Initialising dummy dataset with {length} samples.")
        self.data = torch.randn(length, 1, 28, 28)
        self.labels = torch.randint(0, num_classes, (length,))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]


def load_dataset(batch_size: int = 16, train_ratio: float = 0.8):
    """Create train and test data loaders backed by the dummy dataset."""
    dataset = MouthBreathDataset()
    train_len = int(len(dataset) * train_ratio)
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(
        f"[dataset] Created train loader with {len(train_loader.dataset)} samples "
        f"and test loader with {len(test_loader.dataset)} samples."
    )
    return train_loader, test_loader
