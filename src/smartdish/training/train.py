"""Loop de treino em PyTorch.

Uso:  uv run python -m smartdish.training.train --config configs/default.yaml
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from smartdish.data.dataset import get_dataloader
from smartdish.models.cnn import build_model
from smartdish.utils.common import PROJECT_ROOT, get_device, load_config, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["training"]["device"])
    d, m, t = cfg["data"], cfg["model"], cfg["training"]

    train_loader = get_dataloader(PROJECT_ROOT / d["train_dir"], d["image_size"], d["batch_size"], True, d["num_workers"])
    val_loader = get_dataloader(PROJECT_ROOT / d["val_dir"], d["image_size"], d["batch_size"], False, d["num_workers"])

    model = build_model(m["name"], m["num_classes"], m["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(PROJECT_ROOT / cfg["paths"]["logs"])
    best_acc = 0.0

    for epoch in range(1, t["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc", {"train": tr_acc, "val": va_acc}, epoch)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), Path(ckpt_dir) / f"{m['name']}_best.pt")
    writer.close()
    print(f"Melhor val acc: {best_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
