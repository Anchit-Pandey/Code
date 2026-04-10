"""
train_baseline.py  —  Train CS3T-UNet (paper baseline)
=======================================================
Usage:
    # 5 km/h dataset
    python train_baseline.py --L 1 --train_mat train_adp.mat     --test_mat test_adp.mat
    python train_baseline.py --L 5 --train_mat train_adp.mat     --test_mat test_adp.mat

    # 120 km/h Option B dataset
    python train_baseline.py --L 1 --train_mat train_adp_120B.mat --test_mat test_adp_120B.mat
    python train_baseline.py --L 5 --train_mat train_adp_120B.mat --test_mat test_adp_120B.mat
"""

import os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cs3t_unet  import CS3TUNet, compute_nmse
from dataset    import QuaDRiGaDataset


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_mat',   default='train_adp.mat')
    p.add_argument('--test_mat',    default='test_adp.mat')
    p.add_argument('--results_dir', default='results_baseline')
    p.add_argument('--L',    type=int,   default=1)
    p.add_argument('--T',    type=int,   default=10)
    p.add_argument('--C',    type=int,   default=64)    # C=64 for baseline
    p.add_argument('--epochs',type=int,  default=400)
    p.add_argument('--bs',   type=int,   default=32)
    p.add_argument('--lr',   type=float, default=2e-3)
    p.add_argument('--warmup',type=int,  default=10)
    p.add_argument('--workers',type=int, default=4)
    p.add_argument('--seed', type=int,   default=42)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, training,
              epoch=None, total_epochs=None):
    model.train(training)
    total_loss = total_nmse = n = 0
    n_total = len(loader)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                nmse = compute_nmse(pred.detach(), Y)
            total_loss += loss.item()
            total_nmse += nmse.item()
            n += 1
            if training and n % 100 == 0:
                print(f"    [Ep {epoch}/{total_epochs}] "
                      f"Batch {n}/{n_total} | "
                      f"Loss={total_loss/n:.4e} | "
                      f"NMSE={total_nmse/n:.3f} dB")
    return total_loss / n, total_nmse / n


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.results_dir, exist_ok=True)

    L = args.L
    print(f"\nDevice: {device}")
    print("=" * 60)
    print(f"  Training CS3T-UNet (BASELINE)  |  L={L}")
    print(f"  Epochs={args.epochs}  BS={args.bs}  LR={args.lr}  C={args.C}")
    print("=" * 60)

    # Data
    train_data = QuaDRiGaDataset(args.train_mat, T=args.T, L=L, name='train')
    test_data  = QuaDRiGaDataset(args.test_mat,  T=args.T, L=L, name='test')
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, drop_last=False,  pin_memory=True)

    # Model
    model = CS3TUNet(Nf=64, Nt=64, T=args.T, L=L, C=args.C).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}  ({total_params/1e6:.2f}M)")

    # Optimiser + cosine LR with warmup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        p = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return max(1e-6 / args.lr, 0.5 * (1 + np.cos(np.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_nmse = float('inf')
    best_ckpt     = os.path.join(args.results_dir, f'baseline_L{L}_best.pt')
    last_ckpt     = os.path.join(args.results_dir, f'baseline_L{L}_last.pt')
    history       = dict(train_loss=[], val_loss=[], train_nmse=[], val_nmse=[],
                         lr=[], epoch_time=[])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_nmse = run_epoch(model, train_loader, criterion, optimizer,
                                     device, training=True,
                                     epoch=epoch, total_epochs=args.epochs)
        va_loss, va_nmse = run_epoch(model, test_loader,  criterion, optimizer,
                                     device, training=False)
        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_nmse'].append(tr_nmse)
        history['val_nmse'].append(va_nmse)
        history['lr'].append(lr_now)
        history['epoch_time'].append(elapsed)

        if va_nmse < best_val_nmse:
            best_val_nmse = va_nmse
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_nmse': va_nmse, 'args': vars(args)}, best_ckpt)

        print(f"  Ep {epoch:>4}/{args.epochs} | "
              f"TrLoss={tr_loss:.4e} | TrNMSE={tr_nmse:>8.3f} dB | "
              f"VaNMSE={va_nmse:>8.3f} dB | "
              f"Best={best_val_nmse:>8.3f} dB | "
              f"LR={lr_now:.2e} | {elapsed:.1f}s")

    torch.save({'epoch': args.epochs, 'model_state': model.state_dict(),
                'val_nmse': va_nmse}, last_ckpt)
    with open(os.path.join(args.results_dir, f'history_L{L}.json'), 'w') as f:
        json.dump(history, f)

    print(f"\n  Best val NMSE (L={L}): {best_val_nmse:.3f} dB")
    print(f"  Checkpoints → {args.results_dir}/")


if __name__ == '__main__':
    main()
