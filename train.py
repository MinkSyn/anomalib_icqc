import os
import argparse
from loguru import logger
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from patchcore.model import build_model
from patchcore.datasets import AnoDataset
from config import Config
from tool import verify_device
from const import CardID


class Trainer:
    def __init__(self, cfg, resume):
        self.cfg = cfg
        self.device = verify_device(cfg['device'])
        
        self.cls_card = sorted([cls_name.name for cls_name in CardID])
        
        self.data_root = cfg['data_root']
        self.out_root = os.path.join(cfg['out_root'], f"run{cfg['run']}")
        os.makedirs(self.out_root, exist_ok=True)
        
        self.arch = cfg['feature']['arch']
        self.run_name = f"run{cfg['run']}__{self.arch}__{cfg['info_ver']}"
        
        model = build_model(cfg=self.cfg,
                                 training=True)
        self.model = model.to(self.device)
        
        self.batch_size = cfg['batch_size']
        self.embed_path = os.path.join(cfg['embedding_path'], f"run{cfg['run']}")
        os.makedirs(self.embed_path, exist_ok=True)
        
    def get_loader(self):
        dataloader = {}
        for cls_name in self.cls_card:
            data_dir = os.path.join(self.data_root, f"{cls_name}", 'good')
            if not os.path.exists(data_dir):
                logger.info(f"Not search path for [{cls_name}]")
                continue
            dataset = AnoDataset(data_dir)
            
            dataloader[cls_name] = DataLoader(dataset, 
                                              batch_size=self.batch_size)
            
            logger.info(f"Number of images in {cls_name}: {len(dataloader[cls_name].dataset)}")
        return dataloader
    
    def fit(self):
        dataloader = self.get_loader()
        
        for cls_name in dataloader.keys():
            save_path = os.path.join(self.embed_path, f"{self.run_name}__{cls_name}.pt")
            if os.path.exists(save_path):
                logger.info(f"File already exists: {save_path}")
                continue
            logger.info(f'Embedding {cls_name}')
            
            for idx, batch in enumerate(tqdm(dataloader[cls_name])): 
                images, labels, mask = batch
                images = images.to(self.device)
                embedding = self.model(images, embedding_coreset=None)
                if idx == 0:
                    embedding_coreset = embedding
                else:
                    embedding_coreset = torch.cat((embedding_coreset, embedding), 0)
            torch.save(embedding_coreset, save_path)
            logger.info('-'*20)
        
            
def main(resume, config):
    cfg = Config.load_yaml(config)
    train = Trainer(cfg, resume)
    
    logger.info(f"Data root: {cfg['data_root']}")
    train.fit()
    logger.info(f"Successfully & save path: {cfg['embedding_path']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--resume', '-r', default=False, action='store_true')
    args = parser.parse_args()

    main(resume=args.resume,
         config=args.config)
