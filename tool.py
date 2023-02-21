import os

import torch

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from const import image_transform

def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'

def draw_chart(scores, save_path=None, name_images=None, step=1, pad=2):
    os.makedirs(f'{save_path}', exist_ok=True)
    path_hist = os.path.join(f'{save_path}', f"{name_images}")
    
    bin_start = min(scores[0] + scores[1]) - pad
    bin_end = max(scores[0] + scores[1]) + pad
    bins = np.arange(bin_start, bin_end, step)
    
    plt.figure(figsize=(16, 8))
    plt.hist([scores[0], scores[1]], bins=bins, alpha=1,
                histtype='bar', color=['green', 'blue'], label=['Normal', 'Anomaly'])
    plt.title(f'Histogram score map')
    plt.xlabel('score')
    plt.ylabel('samples')
    plt.legend(loc='upper right')
    plt.savefig(path_hist, transparent=True)
    plt.close()

def binary_classify(image_scores, thresh):
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = 0
    image_classifications[image_classifications >= thresh] = 1
    return image_classifications

def infer_transform(images, device):
    assert len(images) > 0

    transformed_images = []
    for i, image in enumerate(images):
        image = Image.fromarray(image).convert('RGB')
        transformed_images.append(image_transform(image))

    height, width = transformed_images[0].shape[1:3]
    batch = torch.zeros((len(images), 3, height, width))

    for i, transformed_image in enumerate(transformed_images):
        batch[i] = transformed_image

    return batch.to(device)

