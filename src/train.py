import torch
from torch.utils.data import DataLoader
import argparse
from dataset import TimeSeriesDataset
from model import VAE_MLP
from dataset import ATTRIBUTES
import os

def loss_fn(input, recon, mu, logvar, logits, labels):
    mse = torch.nn.functional.mse_loss(input, recon)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input.size(0)
    ce = torch.nn.functional.cross_entropy(logits, labels)
    return mse + kld + ce, mse.item(), kld.item(), ce.item()

def train(args, train_loader, device):
    n_feats = args.w * len(ATTRIBUTES)
    model = VAE_MLP(n_feats, 2, 256, args.z, 2, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    iters = 0
    train_loss, best_loss = 0, float('inf')
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), torch.mode(labels, dim=1).values.to(device).long()
        optimizer.zero_grad()
        recon, mu, logvar, logits = model(data)
        # print("Label min/max:", labels.min().item(), labels.max().item())
        # print("Logits shape:", logits.shape)

        loss, mse_val, kld_val, ce_val = loss_fn(data.view(-1, n_feats), recon.view(-1, n_feats), mu, logvar, logits, labels)
        loss.backward()
        train_loss += loss.item() / args.log
        optimizer.step()

        if iters % args.log == 0:
            torch.save(model.state_dict(), os.path.join(args.o, "weights_latest.pt"))
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(args.o, "weights_best.pt"))
            print(f"Iter {iters}: Total Loss {loss.item():.4f} (MSE: {mse_val:.4f}, KLD: {kld_val:.4f}, CE: {ce_val:.4f})", flush=True)
            train_loss = 0
        iters += 1
        if iters >= args.i:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_ids', nargs="+")
    parser.add_argument('--csv', default="/scratch/gpfs/jl8975/jlanglieb/13_wesad/WESAD/ALL_FROMPKL.csv.gz")
    parser.add_argument('-w', type=int, default=128, help='Window size')
    parser.add_argument('-b', type=int, default=32, help='Batch size')
    parser.add_argument('-i', type=int, default=100000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-z', type=int, default=16, help="Latent dim")
    parser.add_argument('--log', type=int, default=1000, help='Log loss every x iterations')
    parser.add_argument('-o', required=True)
    args = parser.parse_args()

    dataset = TimeSeriesDataset(args.csv, args.subject_ids, args.w)
    train_loader = DataLoader(dataset, batch_size=args.b, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.o, exist_ok=True)

    train(args, train_loader, device)