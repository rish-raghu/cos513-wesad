import argparse
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from scipy.stats import mode
import umap

parser = argparse.ArgumentParser()
parser.add_argument('train_latents')
parser.add_argument('train_labels')
parser.add_argument('test_latents')
parser.add_argument('test_labels')
parser.add_argument('--umap', action='store_true')
# parser.add_argument('--centroid')
args = parser.parse_args()

train_labels = np.load(args.train_labels).astype(int)
train_labels = mode(train_labels, axis=1).mode
mask = ((train_labels <= 4) & (train_labels > 0))
train_labels = train_labels[mask]
train_labels -= 1

train_latents = np.load(args.train_latents)[mask]
if args.umap:
    reducer = umap.UMAP()
    train_latents = reducer.fit_transform(train_latents)

test_labels = np.load(args.test_labels).astype(int)
test_labels = mode(test_labels, axis=1).mode
mask = ((test_labels <= 4) & (test_labels > 0))
test_labels = test_labels[mask]
test_labels -= 1

test_latents = np.load(args.test_latents)[mask]
if args.umap:
    reducer = umap.UMAP()
    test_latents = reducer.fit_transform(test_latents)

# K = len(np.unique(labels))
# clusterer = KMeans(n_clusters=K)
# clusterer = AgglomerativeClustering(n_clusters=K)
# clusterer = SpectralClustering(n_clusters=K)
# cluster_labels = clusterer.fit_predict(latents[mask])
# centers = clusterer.cluster_centers_

# svm = SVC()
# svm.fit(latents[mask], labels[mask])
# cluster_labels = svm.predict(latents[mask])

# score = adjusted_rand_score(labels[mask], cluster_labels)
# print("ARI:", score)

lr = LogisticRegression()
lr.fit(train_latents, train_labels)
train_preds = lr.predict(train_latents)
test_preds = lr.predict(test_latents)

train_acc = (train_labels == train_preds).mean()
test_acc = (test_labels == test_preds).mean()
train_acc_b = balanced_accuracy_score(train_labels, train_preds)
test_acc_b = balanced_accuracy_score(test_labels, test_preds)
print(f"Train Accuracy: {100*train_acc}")
print(f"Test Accuracy: {100*test_acc}")
print(f"Train Accuracy Balanced: {100*train_acc_b}")
print(f"Test Accuracy Balanced: {100*test_acc_b}")
