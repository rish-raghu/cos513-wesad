import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import pyro
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro import poutine
from dataset2 import TimeSeriesDataset, ATTRIBUTES
from train_bayesian_logreg import model


def majority_vote(labels):
    return torch.mode(labels, dim=1).values


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(args.csv, args.subject_ids, args.window, stride=args.stride)
    if len(dataset) == 0:
        print(f"[!] No windows to evaluate for subjects={args.subject_ids} "
              f"with window={args.window}, stride={args.stride}.")
        return

    # Reduce num_workers to 1 to avoid warnings
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=1, pin_memory=True)
    num_classes = int(dataset.labels.max().item() + 1)

    def wrapped_model(x, y=None):
        return model(x, y, num_classes)

    # Block the 'obs' site when creating the guide
    guide = AutoDiagonalNormal(poutine.block(wrapped_model, hide=["obs"]))
    pyro.get_param_store().load(args.weights)
    guide.to(device)
    predictive = Predictive(wrapped_model, guide=guide, num_samples=args.samples)

    # Collect numpy arrays for each batch
    y_true_batches = []
    y_pred_batches = []

    for windows, labels in loader:
        B, W, C = windows.shape
        x = windows.view(B, W * C).to(device)
        y = majority_vote(labels).to(device)

        with torch.no_grad():
            samples = predictive(x)
            
            # Compute logits from sampled weights and biases
            w_samples = samples["w"]  # [num_samples, num_classes, D]
            b_samples = samples["b"]  # [num_samples, num_classes]
            
            # Compute logits for each sample: [num_samples, B, num_classes]
            logits_samples = torch.matmul(x, w_samples.transpose(-1, -2)) + b_samples.unsqueeze(1)
            
            # Average probabilities across samples
            probs = torch.softmax(logits_samples, dim=-1).mean(dim=0)  # [B, num_classes]
            preds = probs.argmax(dim=-1)  # [B]

        # Store batch results as numpy arrays - ensure they are 1D
        y_true_batches.append(y.cpu().numpy().flatten())
        y_pred_batches.append(preds.cpu().numpy().flatten())

    # Print shape info for debugging
    print(f"Number of batches: {len(y_pred_batches)}")
    if len(y_pred_batches) > 0:
        print(f"First batch shape: {y_pred_batches[0].shape}")
    
    # Concatenate all batches
    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)

    print(f"Final predictions shape: {y_pred.shape}")

    os.makedirs(args.o, exist_ok=True)
    np.save(os.path.join(args.o, "y_true.npy"), y_true)
    np.save(os.path.join(args.o, "y_pred.npy"), y_pred)

    acc = (y_true == y_pred).mean()
    from sklearn.metrics import balanced_accuracy_score
    bacc = balanced_accuracy_score(y_true, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Balanced Accuracy: {bacc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to saved pyro param store (e.g. .pt file)")
    parser.add_argument("subject_ids", nargs="+")
    parser.add_argument("--csv", default="/scratch/gpfs/jl8975/jlanglieb/13_wesad/WESAD/ALL_FROMPKL.csv.gz")
    parser.add_argument("-w", "--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("-o", required=True, help="Output prefix or directory")
    args = parser.parse_args()

    if args.stride is None:
        args.stride = args.window

    evaluate(args)