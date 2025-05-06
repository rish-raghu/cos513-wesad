#!/usr/bin/env python
import os
import argparse

import torch
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal

from dataset2 import TimeSeriesDataset, ATTRIBUTES

def model(x_data, y_data, num_classes):
    N, D = x_data.shape
    device = x_data.device  # <- this is crucial

    w = pyro.sample(
        "w", dist.Normal(torch.zeros(num_classes, D, device=device),
                         torch.ones(num_classes, D, device=device)).to_event(2)
    )
    b = pyro.sample(
        "b", dist.Normal(torch.zeros(num_classes, device=device),
                         torch.ones(num_classes, device=device)).to_event(1)
    )

    with pyro.plate("data", N):
        logits = (x_data @ w.T) + b
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y_data)

# Add this function to filter the dataset
def filter_dataset(dataset):
    # Get all labels
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)
    
    all_labels = torch.stack(all_labels)
    # Create mask for labels 1-4
    mask = (all_labels >= 1) & (all_labels <= 4)
    mask = mask.any(dim=1)  # If any label in window is in range
    
    # Create filtered indices
    indices = torch.where(mask)[0].tolist()
    
    # Create subset dataset
    from torch.utils.data import Subset
    filtered_dataset = Subset(dataset, indices)
    
    # Add labels attribute to the Subset for later use
    filtered_dataset.labels = dataset.labels  # Original labels
    
    # You may also want to store the filtered max label
    # Get unique labels from the filtered dataset
    unique_labels = set()
    for i in indices:
        _, label = dataset[i]
        unique_labels.update(label.unique().tolist())
    filtered_dataset.max_label = max(unique_labels)
    
    return filtered_dataset

def train(args):
    # --- prepare data ---
    dataset = TimeSeriesDataset(args.csv, args.subject_ids, args.window, stride=args.stride)
    dataset = filter_dataset(dataset)
    print(f"Filtered dataset size: {len(dataset)}")
    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,      # parallelize loading
        pin_memory=True     # speed up .to(device) transfers
    )
    print('data loaded!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyro.clear_param_store()

    num_classes = int(dataset.labels.max().item() + 1)
    input_dim = args.window * len(ATTRIBUTES)

    # Check if we should load a saved model
    if hasattr(args, 'load_model') and args.load_model:
        if not os.path.exists(args.load_model):
            print(f"Error: Model file {args.load_model} not found!")
            return None
            
        print(f"Loading model from {args.load_model}")
        pyro.get_param_store().load(args.load_model)
        
        # Define wrapped model for the loaded parameters
        def wrapped_model(x, y=None):
            return model(x, y, num_classes)
            
        # For loaded models, we need to block the obs site
        from pyro import poutine
        blocked_model = poutine.block(wrapped_model, hide=["obs"])
        guide = AutoDiagonalNormal(blocked_model)
        guide.to(device)
        
        print("Model loaded successfully!")
        return guide
    
    # --- wrap model with lambda to bind args ---
    def wrapped_model(x, y):
        return model(x, y, num_classes)

    guide = AutoDiagonalNormal(wrapped_model)
    guide.to(device)

    svi = SVI(wrapped_model, guide, Adam({"lr": args.lr}), loss=Trace_ELBO())

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch_idx, (windows, labels) in enumerate(loader):
            windows = windows.to(device)           # [B, W, C]
            B, W, C = windows.shape
            x = windows.view(B, W * C)
            y = labels[:, 0].to(device).long()     # Assume constant label per window

            loss = svi.step(x, y)
            total_loss += loss

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d}  Avg ELBO loss per sample: {avg_loss:.4f}")

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"bayeslogreg_epoch{epoch}.pt")
            pyro.get_param_store().save(ckpt_path)
            print(f"  → Saved guide params to {ckpt_path}")

    return guide



def parse_args():
    p = argparse.ArgumentParser(description="Train Bayesian Logistic Regression on windowed time series")
    p.add_argument("subject_ids", nargs="+", help="List of subject IDs to include")
    p.add_argument("--csv",        default="/scratch/gpfs/jl8975/jlanglieb/13_wesad/WESAD/ALL_FROMPKL.csv.gz")
    p.add_argument("-w", "--window",     type=int, default=128, help="Window size")
    # p.add_argument("-s", "--stride",     type=int, default=1, help="Stride between windows")
    p.add_argument("-s", "--stride",     type=int, default=None, help="Stride between windows (default= window size)")
    p.add_argument("-b", "--batch_size", type=int, default=64)
    p.add_argument("-e", "--epochs",     type=int, default=1)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("-o", "--output_dir", required=True, help="Where to save checkpoints")
    p.add_argument("--ckpt_every",       type=int, default=5, help="Epoch interval to save guide")
    p.add_argument("--load_model", help="Path to saved model to load instead of training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("ATTRIBUTES used:", ATTRIBUTES)
    print("Window size:", args.window, "→ feature dim:", args.window * len(ATTRIBUTES))
    guide = train(args)

    # Example: posterior predictive on a small batch of training data
    # (you can remove this block if you just want the guide saved)
    # dataset = TimeSeriesDataset(args.csv, args.subject_ids, args.window, stride=args.stride)
    # loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # x_batch, y_batch = next(iter(loader))
    # x_flat = x_batch.view(32, -1).to(pyro.get_param_store()._device)
    # predictive = Predictive(model, guide=guide, num_samples=100)
    # samples = predictive(x_flat, None, num_classes)
    # # samples["obs"] has shape [num_samples, batch_size] with int class predictions
    # preds = samples["obs"].mode(0).values  # majority vote across posterior samples
    # print("Posterior class predictions:", preds)