import os

import torch
from PIL import Image
import numpy as np

from model import PatchCore
from tool import verify_device, is_image_file, get_img_transform, binary_classify
from const import AnomalyID


class PatchCoreInference:
    """ Inference after trained and have checkpoint
    
    Args:
        device (str): hardware will use inferernce
        info_path (str): path of checkpoint
    """

    def __init__(self, device='cpu', info_path=None):
        self.device = verify_device(device)
        cfg = self.get_config(info_path)

        model = PatchCore(cfg['hparams'], training=False)
        self.model = model.to(self.device)

        self.threshs = cfg['threshold']
        self.cls_cards = self.threshs.keys()
        self.coresets = cfg['embedding_coreset']
        self.transforms = get_img_transform(cfg['hparams']['img_size'])

    def get_config(self, info_path):
        if not os.path.exists(info_path):
            raise Exception(f'Not exist paths: {info_path}')
        return torch.load(info_path)

    def predict_single(self, input, card_type):
        if card_type not in self.cls_cards:
            raise Exception(f'Not exist class card: {card_type}')

        input = self._preprocess(input)
        _, score = self.model(input, self.coresets[card_type])
        if score >= self.threshs[card_type]:
            return AnomalyID['abnormal'].value
        return AnomalyID['normal'].value

    def predict_batch(self, inputs, card_type, threshold=None):
        if threshold is None:
            threshold = self.threshs[card_type]
        if card_type not in self.cls_cards:
            raise Exception(f'Not exist class card: {card_type}')

        _, scores = self.model(inputs, self.coresets[card_type])
        preds = binary_classify(scores, threshold)
        return preds, scores

    def _preprocess(self, input):
        if isinstance(input, str):
            assert is_image_file(input), f"{input} is not an image."
            with open(input, 'rb') as f:
                input = Image.open(f).convert('RGB')
            assert input is not None, f"{input} is not a valid path."

        if isinstance(input, Image.Image):
            img = self.transforms(input)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            return img
        else:
            raise NotImplementedError