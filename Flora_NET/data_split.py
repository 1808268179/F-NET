import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DataLoaders from an already split dataset (train/val/test)."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Dataset root directory containing train/, val/, and test/.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument(
        "--image_size", type=int, default=224, help="Resize images to image_size x image_size."
    )
    return parser.parse_args()


def build_dataloaders(dataset_root: Path, batch_size: int, num_workers: int, image_size: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    split_dirs = {
        "train": dataset_root / "train",
        "val": dataset_root / "val",
        "test": dataset_root / "test",
    }

    missing = [name for name, path in split_dirs.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing split directories: {missing}. Expected structure: "
            f"{dataset_root}/train, {dataset_root}/val, {dataset_root}/test"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    datasets_by_split = {
        split: datasets.ImageFolder(split_path, transform=transform)
        for split, split_path in split_dirs.items()
    }

    train_classes = datasets_by_split["train"].classes
    for split_name, split_dataset in datasets_by_split.items():
        if split_dataset.classes != train_classes:
            raise ValueError(
                f"Class mismatch in '{split_name}'. "
                f"Expected classes {train_classes}, got {split_dataset.classes}"
            )

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return dataloaders, datasets_by_split


def main() -> None:
    args = parse_args()
    dataloaders, datasets_by_split = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    print(f"Loaded dataset from: {args.dataset_root}")
    print(f"Classes ({len(datasets_by_split['train'].classes)}): {datasets_by_split['train'].classes}")
    for split in ["train", "val", "test"]:
        print(
            f"{split}: {len(datasets_by_split[split])} images, "
            f"{len(dataloaders[split])} batches"
        )


if __name__ == "__main__":
    main()
