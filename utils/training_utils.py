from torch.utils.data import DataLoader, random_split


def get_loaders(dset, batch_size):
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_data, test_data = random_split(dset, [train_size, test_size])
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=dset.collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        shuffle=True,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=dset.collate_fn,
    )
    return train_loader, test_loader
