import os
import json
import argparse
from loguru import logger
from collections import Counter
from tqdm import tqdm

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from patchcore.datasets import AnoDataset
from patchcore.model import build_model
from patchcore.sampling_methods.kcenter import KCenterGreedy
from config import Config
from tool import *
from const import CardID, AnomalyID, nameAno, IMG_SIZE


class Inference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = verify_device(cfg['device'])
        
        self.cls_card = sorted([cls_name.name for cls_name in CardID])
        
        self.data_root = cfg['data_root']
        self.test_root = cfg['test_root']
        self.out_root = os.path.join(cfg['out_root'], f"run{cfg['run']}")
        os.makedirs(self.out_root, exist_ok=True)
        
        self.arch = cfg['feature']['arch']
        self.run_name = f"run{cfg['run']}__{self.arch}__{cfg['info_ver']}"
        
        model = build_model(cfg=self.cfg,
                                 training=False)
        self.model = model.to(self.device)
        
        self.batch_size = cfg['batch_size']
        self.sampling_ratio = cfg['hyper']['ratio']
        self.thresh = cfg['thresh']
        self.embed_path = os.path.join(cfg['embedding_path'], f"run{cfg['run']}")         

    def predict(self, test_root=None, name_folder=None, metric=False, chart=False, coreset=False):
        if test_root is None:
            test_root = self.test_root
        
        os.makedirs(os.path.join(self.out_root, name_folder), exist_ok=True)
        result_path = os.path.join(self.out_root, name_folder, f"{self.run_name}.json")
        
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            logger.info(f'Successfully loaded the file: f"{name_folder}/{self.run_name}.json"')
        else:
            results = {}
            logger.info(f"The name of test folder: {name_folder}")
            for cls_name in self.cls_card:
                if cls_name not in test_root.keys():
                    logger.info(f'Not exits class {cls_name}')
                    continue
                elif coreset:
                    test_dir = {'path': [os.path.join(self.data_root, cls_name, 'good')], 
                                'label': ['normal']}
                else:
                    test_dir = test_root[cls_name]
                    
                embedding_coreset = self.load_coreset(cls_name=cls_name)
                
                outputs = self.predict_card(embedding_coreset, test_dir, cls_name)
                results[cls_name] = outputs
                logger.info('-'*50)

            with open(result_path, 'w') as g:
                json.dump(results, g, indent=4)
                                  
        if chart:
            self.visualize_chart(results, name_folder)
            
        if metric:
            self.visualize_metric(results, name_folder)
            
        return results
 
    def predict_card(self, embedding_coreset, test_dir, cls_name):
        dataloader = self.get_loader(test_dir)
        
        outputs = []
        logger.info(f'Inference: [{cls_name}]')
        for batch in tqdm(dataloader):
            lst_imgs = batch[0].numpy()
            lst_paths = batch[1]
            lst_names = batch[2]
            lst_labels = batch[3]
            ano_batch = infer_transform(lst_imgs, self.device)
            _, anomaly_score = self.model(ano_batch, embedding_coreset)
            pred = binary_classify(anomaly_score, self.thresh[cls_name])
        
            for idx in range(len(lst_imgs)):
                outputs.append({'path': lst_paths[idx],
                                'image': lst_names[idx],
                                'label': nameAno[int(lst_labels[idx])],
                                'pred': nameAno[int(pred[idx])],
                                'prob': round(float(anomaly_score[idx]), 4),
#                                 'map': anomaly_map[idx],
                                })
        return outputs
    
    def get_loader(self, test_dir):
        dataloader = {}

        images = []
        for idx in range(len(test_dir['path'])):
            name_imgs = sorted(os.listdir(test_dir['path'][idx]))
            if test_dir['label'][idx] == 'anomaly':
                label = 1
            elif test_dir['label'][idx] == 'normal':
                label = 0
            else:
                raise Exception(f"Not exist label: {test_dir['label'][idx]}")
            
            logger.info(f"Load data from: {test_dir['path'][idx]}")
            logger.info(f"Label: {test_dir['label'][idx]}, ID: {label}")
            for name in tqdm(name_imgs):
                img_path = os.path.join(test_dir['path'][idx], name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, IMG_SIZE)
                images.append((image, img_path, name, label))
            
        dataloader = DataLoader(images, batch_size=self.batch_size)
        return dataloader 
    
    def visualize_metric(self, results, name_test_dir):
        save_path = os.path.join(self.out_root, name_test_dir, 'ROC_Curve.jpg')
        
        for cls_name in results.keys():
            target, prediction = [], []
            for idx in range(len(results[cls_name])):
                target.append(AnomalyID[results[cls_name][idx]['label']].value)
                prediction.append(AnomalyID[results[cls_name][idx]['pred']].value)
            res_metric = visualize_eval(target, prediction, save_path)
            res_metric['Threshold_config'] = self.threshold[cls_name]
            df = pd.DataFrame.from_dict(res_metric)
            logger.info(f'Metrics of {cls_name}')
            logger.info(df) 
            
    def visualize_chart(self, results_test, name_test_dir):
        results_train = self.predict(name_folder='train_embedded', coreset=True)
        save_path = os.path.join(self.out_root, name_test_dir)

        for cls_name in results_test.keys():
            train_score = [results_train[cls_name][idx]['prob'] for idx in range(len(results_train[cls_name]))]
            test_score = [results_test[cls_name][idx]['prob'] for idx in range(len(results_test[cls_name]))]
            lst_score = [train_score, test_score]
            name_image = f"{cls_name}.jpg"
            draw_chart(scores=lst_score,
                       save_path=save_path,
                       name_images=name_image)
            
    def load_coreset(self, cls_name):
        embedding_path = os.path.join(self.embed_path, f"{self.run_name}__{cls_name}.pt")
        if not os.path.exists(embedding_path):
            raise Exception(f'Not exits embedding of {cls_name}')
        embedding = torch.load(embedding_path)
        
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=self.sampling_ratio)
        coreset = sampler.sample_coreset()
        return coreset
            
            
def main(config, test_root, name_folder, chart, metric):
    cfg = Config.load_yaml(config)
    infer = Inference(cfg)
    
    results = infer.predict(test_root=test_root,
                            name_folder=name_folder,
                            chart=chart,
                            metric=metric)

    for cls_name in results.keys():
        predict = [results[cls_name][idx]['pred'] for idx in range(len(results[cls_name]))]
        view_res = Counter(predict)
        logger.info(f"{cls_name}: {view_res}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', default=None)
    parser.add_argument('--name_folder', default='no_name')
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--chart', '-c', default=False, action='store_true')
    parser.add_argument('--metric', '-m', default=False, action='store_true')
    args = parser.parse_args()

    main(test_root=args.test_root,
         name_folder=args.name_folder,
         chart=args.chart,
         config=args.config,
         metric=args.metric)
