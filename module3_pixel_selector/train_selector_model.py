import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from .selector_model import TinyPatchNet

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "module3_pixel_selector"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_npz_dataset(npz_path: Path):
    data = np.load(npz_path)
    patches = data["patches"]  # N x H x W
    labels = data["labels"]
    X = torch.from_numpy(patches).unsqueeze(1).float()  # N x 1 x H x W
    y = torch.from_numpy(labels).float()
    return TensorDataset(X, y)

def train(args):
    ds = load_npz_dataset(Path(args.data))
    val_size = max(int(len(ds) * 0.1), 16)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TinyPatchNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # val
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                preds = (out > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            save_path = DEFAULT_MODEL_DIR / "best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "module3" / "sample.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    cli()
