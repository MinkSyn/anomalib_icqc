import os

import torch
from PIL import Image
import numpy as np

from model import PatchcoreModel
from tool import verify_device, is_image_file, get_img_transform, binary_classify
from const import AnomalyID


class AnoInference:
    def __init__(self, device='cpu', info_path=None):
        self.device = verify_device(device)
        self.cfg = self.get_config(info_path)
        cfg_model = self.cfg['cfg_model']
        
        model = PatchcoreModel(input_size=cfg_model['img_size'],
                               layers=cfg_model['backbone']['layer'],
                               arch=cfg_model['backbone']['arch'],
                               pre_trained=cfg_model['backbone']['pretrained'],
                               num_neighbors=cfg_model['nn_search']['num_neighbors'],
                               method_dis=cfg_model['nn_search']['distance'],
                               training=False,
                               )
        self.model = model.to(self.device)
        
        self.threshs = self.cfg['threshold']
        self.cls_cards = self.threshs.keys()
        self.coresets = self.cfg['embedding_coreset']
        self.transforms = get_img_transform(cfg_model['img_size']) 
        
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
        
        inputs = self._preprocess(inputs)
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
        elif isinstance(input, list):
            imgs = input.to(self.device)
            return imgs
            # imgs = [self._preprocess(i) for i in input]
            # return torch.cat(imgs, dim=0)
        else:
            raise NotImplementedError