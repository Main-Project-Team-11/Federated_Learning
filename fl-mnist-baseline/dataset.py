import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_ROOT = Path(__file__).resolve().parent / "data"


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    include_test: bool = True,
):
    """
    Loads MNIST, splits the TRAIN set into `num_partitions` equal, non-overlapping
    chunks, and returns the chunk belonging to `partition_id` (i.e. this client's
    private data). Returns: train_loader, val_loader, and optionally test_loader.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id must be in [0, num_partitions)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

    # Split the 60,000 training images into num_partitions equal chunks
    partition_size = len(full_train) // num_partitions
    lengths = [partition_size] * num_partitions
    lengths[-1] += len(full_train) - sum(lengths)  # give remainder to last client
    partitions = random_split(full_train, lengths, generator=torch.Generator().manual_seed(42))

    client_data = partitions[partition_id]

    # Within this client's data, carve out 10% for local validation
    val_size = int(0.1 * len(client_data))
    train_size = len(client_data) - val_size
    train_subset, val_subset = random_split(
        client_data, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = None
    if include_test:
        full_test = datasets.MNIST(
            root=DATA_ROOT, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(full_test, batch_size=batch_size)

    return train_loader, val_loader, test_loader
