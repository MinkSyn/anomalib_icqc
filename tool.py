import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from const import AnomalyID


def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
def is_image_file(filename: str):
    return filename.lower().endswith(IMG_EXTENSIONS)

def get_img_transform(img_size):
    transform = T.Compose([T.Resize(img_size),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
    return transform

def binary_classify(image_scores, thresh):
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = AnomalyID['normal'].value
    image_classifications[image_classifications >= thresh] = AnomalyID['abnormal'].value
    return image_classifications

def draw_chart(scores, save_path=None, name_images=None, step=1, pad=2):
    os.makedirs(f'{save_path}', exist_ok=True)
    path_hist = os.path.join(f'{save_path}', f"{name_images}")
    
    bin_start = min(scores[0] + scores[1]) - pad
    bin_end = max(scores[0] + scores[1]) + pad
    bins = np.arange(bin_start, bin_end, step)
    
    plt.figure(figsize=(16, 8))
    plt.hist([scores[0], scores[1]], bins=bins, alpha=1,
                histtype='bar', color=['green', 'blue'], label=['Normal', 'Abnormal'])
    plt.title(f'Histogram score map')
    plt.xlabel('score')
    plt.ylabel('samples')
    plt.legend(loc='upper right')
    plt.savefig(path_hist, transparent=True)
    plt.close()

def visualize_eval(target, prediction, prob, save_path):
    score = roc_auc_score(target, prediction)
    precision, recall, _ = optimal_threshold(target, prediction)
    _, _, threshold = optimal_threshold(target, prob)
    res = {
        'ROC': score,
        'Precision': precision,
        'Recall': recall, 
        'Threshold_auto': threshold,
    }
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

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
    target = np.array(target)
    prediction = np.array(prediction)
    
    precision, recall, thresholds = precision_recall_curve(target.flatten(), prediction.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    idx = np.argmax(f1)
    return precision[idx], recall[idx], thresholds[idx]