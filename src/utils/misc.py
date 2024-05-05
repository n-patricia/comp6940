
import numpy as np
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F



def get_representations(model, dataloader, device):
    model.eval()

    outputs = []
    intermediates = []
    labels = []
    with torch.no_grad():
        for (x,y) in tqdm(dataloader):
            x = x.to(device)
            y_pred, h = model(x)
            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def get_tsne(data, n_components=2, n_images=None):
    if n_images is not None:
        data = data[:n_images]
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data



