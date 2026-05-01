"""Train a conditional GAN to generate fake potato leaf images.

Default Kaggle dataset layout:

    /kaggle/input/potato-disease-leaf-datasetpld/PLD_3_Classes_256/
        Training/
            Early_Blight/
            Healthy/
            Late_Blight/
        Validation/
        Testing/

The script trains on the Training split by default and writes generated images,
checkpoints, and class metadata to /kaggle/working/gan_outputs on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_CANDIDATES = [
    Path("/kaggle/input/potato-disease-leaf-datasetpld/PLD_3_Classes_256"),
    Path("/kaggle/input/potato-disease-leaf-dataset-pld/PLD_3_Classes_256"),
    Path("/kaggle/input/potato-disease-leaf-dataset/PLD_3_Classes_256"),
    Path("data") / "PlantVillage",
]
DEFAULT_OUTPUT_DIR = (
    Path("/kaggle/working/gan_outputs")
    if Path("/kaggle/working").exists()
    else Path("outputs") / "gan_outputs"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_default_input_dir() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[0]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_class_dirs(input_dir: Path, split: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    search_roots = [input_dir / split] if split != "all" else [
        input_dir / "Training",
        input_dir / "Validation",
        input_dir / "Testing",
    ]

    class_dirs: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and any(is_image_file(path) for path in child.iterdir()):
                class_dirs.setdefault(child.name, child)

    if not class_dirs:
        # Fallback for datasets arranged directly as class folders.
        for child in sorted(input_dir.iterdir()):
            if child.is_dir() and any(is_image_file(path) for path in child.iterdir()):
                class_dirs.setdefault(child.name, child)

    if not class_dirs:
        raise ValueError(f"No class image folders found under: {input_dir}")

    return [class_dirs[name] for name in sorted(class_dirs)]


class PotatoLeafDataset(Dataset):
    def __init__(self, input_dir: Path, split: str, image_size: int):
        self.class_dirs = find_class_dirs(input_dir, split)
        self.class_names = [class_dir.name for class_dir in self.class_dirs]
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []

        if split == "all":
            roots = [input_dir / "Training", input_dir / "Validation", input_dir / "Testing"]
            for root in roots:
                if not root.exists():
                    continue
                for class_name in self.class_names:
                    class_dir = root / class_name
                    if class_dir.exists():
                        self.samples.extend(
                            (path, self.class_to_idx[class_name])
                            for path in sorted(class_dir.iterdir())
                            if is_image_file(path)
                        )
        else:
            for class_dir in self.class_dirs:
                label = self.class_to_idx[class_dir.name]
                self.samples.extend(
                    (path, label) for path in sorted(class_dir.iterdir()) if is_image_file(path)
                )

        if not self.samples:
            raise ValueError(f"No images found for split '{split}'")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, label


class Generator(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, embedding_dim: int, channels: int = 3):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        input_dim = noise_dim + embedding_dim

        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_features = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        model_input = torch.cat([noise, label_features], dim=1)
        return self.net(model_input)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, image_size: int, channels: int = 3):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = images.shape
        label_map = self.label_embedding(labels).view(batch_size, 1, image_height, image_width)
        model_input = torch.cat([images, label_map], dim=1)
        return self.net(model_input).view(-1)


def weights_init(module: nn.Module) -> None:
    class_name = module.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def save_class_samples(
    generator: Generator,
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    noise_dim: int,
    images_per_class: int,
    epoch: int | str,
) -> None:
    generator.eval()
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        all_images = []
        for class_index, class_name in enumerate(class_names):
            labels = torch.full((images_per_class,), class_index, dtype=torch.long, device=device)
            noise = torch.randn(images_per_class, noise_dim, 1, 1, device=device)
            fake_images = generator(noise, labels)
            save_image(
                fake_images,
                sample_dir / f"epoch_{epoch}_{class_name}.png",
                normalize=True,
                nrow=images_per_class,
            )
            all_images.append(fake_images.cpu())

        grid = make_grid(torch.cat(all_images, dim=0), nrow=images_per_class, normalize=True)
        save_image(grid, sample_dir / f"epoch_{epoch}_all_classes.png")

    generator.train()


def train_gan(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = PotatoLeafDataset(args.input_dir, args.split, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    if len(dataloader) == 0:
        raise ValueError("Dataset is smaller than batch size. Lower --batch-size and retry.")

    generator = Generator(
        noise_dim=args.noise_dim,
        num_classes=len(dataset.class_names),
        embedding_dim=args.embedding_dim,
    ).to(device)
    discriminator = Discriminator(
        num_classes=len(dataset.class_names),
        image_size=args.image_size,
    ).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    metadata = {
        "input_dir": str(args.input_dir),
        "split": args.split,
        "class_names": dataset.class_names,
        "num_images": len(dataset),
        "image_size": args.image_size,
        "noise_dim": args.noise_dim,
    }
    (output_dir / "class_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Device: {device}")
    print(f"Classes: {dataset.class_names}")
    print(f"Training images: {len(dataset)}")
    print(f"Output directory: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        running_d_loss = 0.0
        running_g_loss = 0.0

        for real_images, labels in dataloader:
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            discriminator.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, labels)
            real_loss = criterion(real_logits, real_targets)

            noise = torch.randn(batch_size, args.noise_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, len(dataset.class_names), (batch_size,), device=device)
            fake_images = generator(noise, fake_labels)
            fake_logits = discriminator(fake_images.detach(), fake_labels)
            fake_loss = criterion(fake_logits, fake_targets)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            generator.zero_grad(set_to_none=True)
            fake_logits = discriminator(fake_images, fake_labels)
            g_loss = criterion(fake_logits, real_targets)
            g_loss.backward()
            optimizer_g.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        print(f"Epoch [{epoch}/{args.epochs}] D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}")

        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            save_class_samples(
                generator=generator,
                class_names=dataset.class_names,
                output_dir=output_dir,
                device=device,
                noise_dim=args.noise_dim,
                images_per_class=args.generate_per_class,
                epoch=epoch,
            )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "class_names": dataset.class_names,
                    "args": vars(args),
                },
                checkpoint_dir / f"gan_epoch_{epoch}.pth",
            )

    save_class_samples(
        generator=generator,
        class_names=dataset.class_names,
        output_dir=output_dir,
        device=device,
        noise_dim=args.noise_dim,
        images_per_class=args.generate_per_class,
        epoch="final",
    )
    print("Done. Fake images are in:", output_dir / "samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional GAN for potato leaf images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=resolve_default_input_dir(),
        help="Path to PLD_3_Classes_256. Default: Kaggle PLD path if available.",
    )
    parser.add_argument(
        "--split",
        choices=("Training", "Validation", "Testing", "all"),
        default="Training",
        help="Dataset split to train on. Default: Training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where outputs will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=64, help="GAN image size. Use 64 for DCGAN.")
    parser.add_argument("--noise-dim", type=int, default=100, help="Latent noise vector size.")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Class embedding size.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--sample-every", type=int, default=5, help="Save samples every N epochs.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--generate-per-class", type=int, default=8, help="Fake images saved per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {' '.join(unknown_args)}")
    return args


def main() -> None:
    args = parse_args()

    if args.image_size != 64:
        raise ValueError("This DCGAN architecture expects --image-size 64.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.batch_size <= 1:
        raise ValueError("--batch-size must be greater than 1.")
    if args.sample_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("--sample-every and --checkpoint-every must be greater than 0.")

    train_gan(args)


if __name__ == "__main__":
    main()
