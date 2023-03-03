import os
import json
import argparse
from loguru import logger
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from components.kcenter import KCenterGreedy
from model import PatchcoreModel
from datasets import AnoDataset
from config import Config
from tool import verify_device, get_img_transform, optimal_threshold
from const import CardID


class Embedding:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = verify_device(cfg['device'])
        
        self.cls_cards = sorted([card_type.name for card_type in CardID])
        
        self.data_root = cfg['data_root']
        self.test_root = cfg['test_root']
        
        self.arch = cfg['hparams']['backbone']['arch']
        self.run_name = f"run{cfg['run']}__{self.arch}"
        
        self.icqc2ano = cfg['icqc2ano']
        
        self.hparams = cfg['hparams']
        model = PatchcoreModel(input_size=self.hparams['img_size'],
                               layers=self.hparams['backbone']['layer'],
                               arch=self.hparams['backbone']['arch'],
                               pre_trained=self.hparams['backbone']['pretrained'],
                               num_neighbors=self.hparams['nn_search']['num_neighbors'],
                               method_dis=self.hparams['nn_search']['distance'],
                               training=True,
                               )
        self.model = model.to(self.device)
        
        self.transforms = get_img_transform(cfg['hparams']['img_size'])
        self.hp_core = cfg['hparams']['coreset']
        self.batch_size = cfg['hparams']['batch_size']
        
        self.embed_path = cfg['embedding_path']
        self.out_root = os.path.join(cfg['out_root'], f"run{cfg['run']}")
        os.makedirs(self.out_root, exist_ok=True)   
        
        self.data_train = self.get_loader(split='train')
        self.data_test = self.get_loader(split='test')
        
        self.threshs = {}
        self.coresets = {}
        
    def get_loader(self, split):
        dataloader, data_info = {}, {}
        logger.info(f"Number of images in {split} dataset")
        for card_type in self.cls_cards:
            if split == 'train':
                data_dir = os.path.join(self.data_root, f"{card_type}")
            elif split == 'test':
                data_dir = os.path.join(self.test_root, f"{card_type}")
                
            if not os.path.exists(data_dir):
                data_info[card_type] = None
                continue
            dataset = AnoDataset(root=data_dir,
                                 split=split,
                                 icqc2ano=self.icqc2ano,
                                 transforms=self.transforms)
            
            dataloader[card_type] = DataLoader(dataset, 
                                              batch_size=self.batch_size
                                              )
            data_info[card_type] = len(dataloader[card_type].dataset)
        print(pd.DataFrame({'samples': data_info.values()}, index=data_info.keys()))
        return dataloader
    
    def embedding_coreset(self):        
        for card_type in self.data_train.keys():
            logger.info(f'Embedding {card_type}')
            
            for idx, batch in enumerate(tqdm(self.data_train[card_type])): 
                images, _, _ = batch
                images = images.to(self.device)
                embedding = self.model(images, embedding_coreset=None)
                if idx == 0:
                    embedding_vectors = embedding
                else:
                    embedding_vectors = torch.cat((embedding_vectors, embedding), 0)
                    
            self.coresets[card_type] = self.subsample_embedding(embedding_vectors)
            logger.info('-'*20)
    
    def subsample_embedding(self, embedding):
        sampler = KCenterGreedy(embedding=embedding, 
                                sampling_ratio=self.hp_core['ratio'],
                                eps=self.hp_core['eps'],
                                seed=self.hp_core['seed'])
        coreset = sampler.sample_coreset()
        return coreset
            
    def find_threshold(self):
        self.model.training = False
        
        logger.info(f'Inference train & test')
        for card_type in self.data_test.keys():
            logger.info(f"Class card: {card_type}")
            result_train = self.inference_one_card(self.data_train[card_type], card_type)
            result_test = self.inference_one_card(self.data_test[card_type], card_type)
            
            all_results = result_train + result_test
            targets, scores = [], []
            for idx in range(len(all_results)):
                targets.append(all_results[idx]['label'])
                scores.append(all_results[idx]['score'])
                
            _, _, threshold = optimal_threshold(targets, scores)
            self.threshs[card_type] = threshold
            
            result_path = os.path.join(self.out_root, f"{self.run_name}__{card_type}.pt")
            save_results = {'train': result_train,
                            'test': result_test}
            torch.save(save_results, result_path)     
            
    def inference_one_card(self, dataloader, card_type):
        output = []
        for batch in tqdm(dataloader):
            lst_imgs, lst_paths, lst_targets = batch
            lst_imgs = lst_imgs.to(self.device)
            _, scores = self.model(lst_imgs, self.coresets[card_type])
            
            for idx in range(len(lst_imgs)):
                output.append({'path': lst_paths[idx],
                               'label': lst_targets[idx],
                               'score': round(float(scores[idx]), 4),
                               })
        return output
        
    def fit(self):
        # Create embedding coreset and save into self.coresets
        self.embedding_coreset()
        
        # Find best threshold and save into self.threshs
        self.find_threshold()
        
        # Save config 
        save_path = os.path.join(self.embed_path, f"{self.run_name}.pt")
        info_ver = {
            'run': self.cfg['run'],
            'cfg_model': {
                'img_size': self.hparams['img_size'],
                'backbone': {
                    'layer': self.hparams['backbone']['layer'],
                    'arch': self.hparams['backbone']['arch'],
                    'pretrained': self.hparams['backbone']['pretrained'],
                },
                'nn_search': {
                    'num_neighbors': self.hparams['nn_search']['num_neighbors'],
                    'distance': self.hparams['nn_search']['distance'],
                },
            },
            'threshold': self.threshs,
            'embedding_coreset': self.coresets,
            }
        torch.save(info_ver, save_path)   
                    
                    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    
    cfg = Config.load_yaml(args.config)
    train = Embedding(cfg)
    
    logger.info(f"Data root: {cfg['data_root']}")
    train.fit()
    logger.info(f"Successfully & save path: {cfg['embedding_path']}")

if __name__ == '__main__':
    main()
