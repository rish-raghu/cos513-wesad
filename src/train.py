import torch
from torch.utils.data import DataLoader
import argparse
from dataset import TimeSeriesDataset
from model import VAE_MLP
from dataset import ATTRIBUTES

def loss_fn(input, output, mu, logvar):
    mse = torch.nn.functional.mse_loss(input, output)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

def train(args, train_loader, device):
    n_feats = args.w * len(ATTRIBUTES)
    model = VAE_MLP(n_feats, 2, 256, 16, 2, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    iters = 0
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_fn(data.view(-1, n_feats), recon.view(-1, n_feats), mu, logvar)
        loss.backward()
        train_loss += loss.item() / args.log
        optimizer.step()

        if iters % args.log == 0:
            print(f"Iter {iters}: Loss {loss.item()}")
            train_loss = 0
        iters +=1
        if iters >= args.i: break

    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default="/scratch/gpfs/jl8975/jlanglieb/13_wesad/WESAD/ALL_FROMPKL.csv.gz")
    parser.add_argument('-w', type=int, default=128, help='Window size')
    parser.add_argument('-b', type=int, default=32, help='Batch size')
    parser.add_argument('-i', type=int, default=100000, help='Number of iterations')
    parser.add_argument('-log', type=int, default=1000, help='Log loss every x iterations')
    args = parser.parse_args()

    dataset = TimeSeriesDataset(args.csv, args.w)
    train_loader = DataLoader(dataset, batch_size=args.b, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(args, train_loader, device)
