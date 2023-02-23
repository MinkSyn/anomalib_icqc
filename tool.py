import os
from typing import Tuple, Any

import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from const import image_transform

def allowed_file(filename):
    return ("." in filename and filename.rsplit(".", 1)[1].lower() in ["png", "jpg", "jpeg"])

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

def visualize_eval(target, prediction, save_path):
    # score = roc_auc_score(target, prediction)
    target = np.array(target)
    prediction = np.array(prediction)
    precision, recall, threshold = optimal_threshold(target, prediction)
    res = {
        # 'ROC': score,
        'Precision': precision,
        'Recall': recall, 
        'Threshold_auto': threshold,
    }
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

    fpr, tpr, thresholds = roc_curve(target, prediction)
    axes[0].plot(fpr, tpr)
    axes[0].title.set_text('ROC Curve (tpr-fpr)')

    axes[1].plot(thresholds, fpr)
    axes[1].plot(thresholds, tpr, color='red')
    axes[1].axvline(x=threshold, color='yellow')
    axes[1].grid()
    axes[1].title.set_text('fpr/tpr - thresh')

    plt.savefig(save_path, transparent=True)
    plt.close()
    return res


def optimal_threshold(target, prediction):
    precision, recall, thresholds = precision_recall_curve(target.flatten(), prediction.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    idx = np.argmax(f1)
    return precision[idx], recall[idx], thresholds[idx]