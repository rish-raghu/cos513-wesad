import torch
from torch.utils.data import DataLoader
import argparse
from dataset import TimeSeriesDataset
from model import VAE_MLP, TimeVAE
from dataset import ATTRIBUTES
import os

MODEL_TYPE="time"

def loss_fn(input, output, mu, logvar):
    mse = torch.nn.functional.mse_loss(input, output)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

def train(args, train_loader, device):
    n_feats = args.w * len(ATTRIBUTES)
    if MODEL_TYPE=="base":
        model = VAE_MLP(n_feats, 2, 256, args.z, 2, 256).to(device)
    elif MODEL_TYPE=="time":
        model = TimeVAE(args.w, len(ATTRIBUTES), 3, 256, 3, args.z, 3, 256, args.p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    iters = 0
    train_loss, best_loss = 0, float('inf')
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_fn(data.view(-1, n_feats), recon.view(-1, n_feats), mu, logvar)
        #recon, mu = model(data)
        #loss = loss_fn(data.view(-1, n_feats), recon.view(-1, n_feats))
        loss.backward()
        train_loss += loss.item() / args.log
        optimizer.step()

        if iters % args.log == 0:
            torch.save(model.state_dict(), os.path.join(args.o, "weights_latest.pt"))
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(args.o, "weights_best.pt"))
            print(f"Iter {iters}: Loss {loss.item()}", flush=True)
            train_loss = 0
        iters +=1
        if iters >= args.i: break

    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_ids', nargs="+")
    parser.add_argument('--csv', default="/scratch/gpfs/jl8975/jlanglieb/13_wesad/WESAD/ALL_FROMPKL.csv.gz")
    parser.add_argument('-w', type=int, default=128, help='Window size')
    parser.add_argument('-b', type=int, default=32, help='Batch size')
    parser.add_argument('-i', type=int, default=100000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-z', type=int, default=16, help="Latent dim")
    parser.add_argument('-p', type=int, default=None, help='Poly trend degree')
    parser.add_argument('--log', type=int, default=1000, help='Log loss every x iterations')
    parser.add_argument('-o', required=True)
    args = parser.parse_args()

    dataset = TimeSeriesDataset(args.csv, args.subject_ids, args.w)
    train_loader = DataLoader(dataset, batch_size=args.b, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.o, exist_ok=True)

    train(args, train_loader, device)
