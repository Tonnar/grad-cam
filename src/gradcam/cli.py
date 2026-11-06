"""Command-line interface: train, predict, explain and sanity-check.

    gradcam train   --data data/pets --out runs/densenet
    gradcam predict --checkpoint runs/densenet/best.pt photo.jpg
    gradcam explain --checkpoint runs/densenet/best.pt photo.jpg --out heatmap.png
    gradcam sanity  --checkpoint runs/densenet/best.pt photo.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gradcam.cam import GradCAM, overlay
from gradcam.data import load_image, make_loaders
from gradcam.model import TARGET_LAYERS, build_classifier, load_checkpoint, save_checkpoint
from gradcam.sanity import randomization_test


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate(model: nn.Module, loader, device, n_classes: int) -> tuple[float, np.ndarray]:
    """Accuracy and confusion matrix (rows: true class, columns: predicted)."""
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            np.add.at(confusion, (y.numpy(), pred.numpy()), 1)
    return float(np.trace(confusion) / max(1, confusion.sum())), confusion


def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = _device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, classes = make_loaders(
        args.data, args.val_fraction, args.batch_size, args.image_size, args.seed, tuple(args.exclude), args.workers
    )
    hidden = tuple(args.hidden)
    model = build_classifier(len(classes), args.arch, not args.no_pretrained, not args.finetune, hidden).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        val_acc, _ = evaluate(model, val_loader, device, len(classes))
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d}  loss {loss_sum / total:.4f}  train {correct / total:.3f}  val {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            save_checkpoint(out / "best.pt", model, classes, args.arch, hidden, image_size=args.image_size)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no improvement for {args.patience} epochs; stopping")
                break

    with (out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    model, _ = load_checkpoint(out / "best.pt", device)
    acc, confusion = evaluate(model, val_loader, device, len(classes))
    per_class = {c: float(confusion[i, i] / max(1, confusion[i].sum())) for i, c in enumerate(classes)}
    results = {"val_accuracy": acc, "per_class_accuracy": per_class, "classes": classes,
               "confusion_matrix": confusion.tolist(), "epochs_run": len(history)}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"val_accuracy": acc, "per_class_accuracy": per_class}, indent=2))
    return results


def predict(args) -> list[dict]:
    model, ckpt = load_checkpoint(args.checkpoint, _device(args.device))
    device = next(model.parameters()).device
    rows = []
    for path in args.images:
        x, _ = load_image(path, ckpt.get("image_size", 224))
        with torch.no_grad():
            probs = model(x.to(device)).softmax(1)[0].cpu()
        top = probs.topk(min(args.top_k, len(probs)))
        rows.append({"image": str(path),
                     "top": [(ckpt["classes"][i], round(float(p), 4)) for p, i in zip(top.values, top.indices)]})
        print(rows[-1])
    return rows


def explain(args) -> Path:
    from PIL import Image

    model, ckpt = load_checkpoint(args.checkpoint, _device(args.device))
    device = next(model.parameters()).device
    classes = ckpt["classes"]
    x, rgb = load_image(args.image, ckpt.get("image_size", 224))
    target = classes.index(args.target) if args.target else None
    with GradCAM(model, args.layer or TARGET_LAYERS[ckpt["arch"]]) as cam:
        heatmap, logits = cam(x.to(device), target)
    pred = classes[int(logits.argmax(1))]
    explained = args.target or pred
    out = Path(args.out or Path(args.image).with_suffix("").as_posix() + f"_gradcam_{explained}.png")
    Image.fromarray(overlay(rgb, heatmap[0].cpu().numpy(), args.alpha)).save(out)
    print(f"predicted {pred}; explained {explained}; saved {out}")
    return out


def sanity(args) -> list[dict]:
    model, ckpt = load_checkpoint(args.checkpoint, _device(args.device))
    device = next(model.parameters()).device
    x = torch.cat([load_image(p, ckpt.get("image_size", 224))[0] for p in args.images]).to(device)
    rows = randomization_test(model, args.layer or TARGET_LAYERS[ckpt["arch"]], x, steps=args.steps)
    for r in rows:
        print(f"step {r['step']}: randomised down to {r['randomized_down_to']:<28} spearman {r['spearman']:+.3f}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gradcam", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a classifier on an image folder")
    t.add_argument("--data", required=True, help="folder with one sub-folder per class")
    t.add_argument("--out", default="runs/latest")
    t.add_argument("--arch", default="densenet121", choices=list(TARGET_LAYERS))
    t.add_argument("--hidden", type=int, nargs="*", default=[512], help="hidden sizes of the classifier head")
    t.add_argument("--finetune", action="store_true", help="train the backbone too (default: head only)")
    t.add_argument("--no-pretrained", action="store_true", help="random initial weights (for tests)")
    t.add_argument("--exclude", nargs="*", default=[], help="class folders to ignore")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--image-size", type=int, default=224)
    t.add_argument("--val-fraction", type=float, default=0.2)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=train)

    for name, func, helptext in [("predict", predict, "top-k predictions"),
                                 ("sanity", sanity, "model-randomisation sanity check")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("images", nargs="+")
        s.set_defaults(func=func)
        if name == "predict":
            s.add_argument("--top-k", type=int, default=3)
        else:
            s.add_argument("--steps", type=int, default=5)
            s.add_argument("--layer", help="target layer (default: last conv block)")

    e = sub.add_parser("explain", help="save a Grad-CAM overlay")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("image")
    e.add_argument("--target", help="class name to explain (default: predicted class)")
    e.add_argument("--layer", help="target layer (default: last conv block)")
    e.add_argument("--alpha", type=float, default=0.4)
    e.add_argument("--out")
    e.set_defaults(func=explain)

    for s in sub.choices.values():
        s.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
