import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from dataset import TimeSeriesDataset
from dataset import ATTRIBUTES
from model import VAE_MLP
from train import loss_fn
import os

def eval(args, model, test_loader, device):
    model.eval()

    # eval_loss = 0
    mus = torch.zeros((len(test_loader.dataset), args.z))
    logvars = torch.zeros_like(mus)
    labels = torch.zeros((mus.shape[0], args.w))
    for batch_idx, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(device)
            recon, mu, logvar = model(data)
            #recon, mu = model(data)

            sl = slice(batch_idx*args.b, (batch_idx+1)*args.b)
            mus[sl], logvars[sl], labels[sl] = mu.cpu(), logvar.cpu(), label        
            #mus[sl], labels[sl] = mu.cpu(), label     
    
    return mus, logvars, labels
    #return mus, labels


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('subject_ids', nargs="+")
    parser.add_argument('--csv', default="/scratch/gpfs/jl8975/jlanglieb/13_wesad/WESAD/ALL_FROMPKL.csv.gz")
    parser.add_argument('-w', type=int, default=128, help='Window size')
    parser.add_argument('-b', type=int, default=32, help='Batch size')
    parser.add_argument('-z', type=int, default=16)
    #parser.add_argument('--log', type=int, default=1000, help='Log loss every x iterations')
    parser.add_argument('-o', required=True)
    args = parser.parse_args()

    dataset = TimeSeriesDataset(args.csv, args.subject_ids, args.w, stride=args.w)
    test_loader = DataLoader(dataset, batch_size=args.b, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = args.w * len(ATTRIBUTES)
    model = VAE_MLP(n_feats, 2, 256, args.z, 2, 256).to(device)
    model.load_state_dict(torch.load(args.weights, weights_only=True))

    mus, logvars, labels = eval(args, model, test_loader, device)
    #mus, labels = eval(args, model, test_loader, device)
    np.save(args.o + "_mu.npy", mus.numpy())
    np.save(args.o + "_logvar.npy", logvars.numpy())
    np.save(args.o + "_label.npy", labels.numpy())
    print("Finished")
