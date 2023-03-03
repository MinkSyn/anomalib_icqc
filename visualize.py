import os
import time
from tqdm import tqdm
import argparse
from loguru import logger

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from tool import verify_device, visualize_eval, draw_chart, get_img_transform
from const import AnomalyID
from datasets import AnoDataset
from inference import AnoInference


class VisualAno:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = verify_device(cfg['device'])
        
        self.data_root = cfg['data_root']
        self.test_root = cfg['test_root']
        self.out_root = os.path.join(cfg['out_root'], f"run{cfg['run']}")
        
        self.arch = cfg['hparams']['backbone']['arch']
        self.run_name = f"run{cfg['run']}__{self.arch}"
        
        self.icqc2ano = cfg['icqc2ano']
        self.batch_size = cfg['hparams']['batch_size']
        self.transforms = get_img_transform(cfg['hparams']['img_size'])
        
        self.embedded_path = os.path.join(cfg['embedding_path'], f"{self.run_name}.pt")
        ckpt = self.get_config(self.embedded_path)
        self.threshs = ckpt['threshold']
        self.coresets = ckpt['embedding_coreset']
        self.cls_cards = self.threshs.keys()
        
        self.infer = AnoInference(self.device, self.embedded_path)
        self.results_train, self.results_test = self.get_results()
        
    def get_config(self, info_path):
        if not os.path.exists(info_path):
            raise Exception(f'Not exist paths: {info_path}')
        return torch.load(info_path)
    
    def get_results(self):
        results_train, results_test= {}, {}
        for card_type in self.cls_cards:
            result_path = os.path.join(self.out_root, f"{self.run_name}__{card_type}.pt")
            if not os.path.exists(result_path):
                raise Exception(f"Not exist result path: {result_path}")
            all_results = torch.load(result_path)
            
            results_train[card_type] = self.append_prediction(all_results['train'], card_type)
            results_test[card_type] = self.append_prediction(all_results['test'], card_type)
        return results_train, results_test
            
    def append_prediction(self, datasets, card_type):
        for idx in range(len(datasets)):
            score = datasets[idx]['score']
            if score >= self.threshs[card_type]:
                pred = AnomalyID['abnormal'].value
            else:
                pred = AnomalyID['normal'].value
            datasets[idx]['pred'] = pred
        return datasets
    
    def check_gpu(self, num_samples):
        for card_type in self.cls_cards:
            if num_samples > len(self.results_train[card_type]):
                num_samples = len(self.results_train[card_type])
                
            time_infer, gpu_memory = [], [],
            logger.info(f"Check gpu for {card_type}")
            for idx in range(num_samples):
                img_path = self.results_train[card_type][idx]['path']
                start_time = time.perf_counter()
                _ = self.infer.predict_single(img_path, card_type)
                time_infer.append(time.perf_counter() - start_time)
                gpu_memory.append(torch.cuda.memory_allocated())
            logger.info(f"GPU memory: {sum(gpu_memory) / (1024**3 * num_samples)} GB")
            logger.info(f"Time inference: {sum(time_infer)*100 / num_samples} ms")
            logger.info("-"*50)
            
    def inference_test(self, test_root):
        outputs = {}
        for card_type in self.cls_cards:
            test_dir = os.path.join(test_root, card_type)
            
            if not os.path.exists(test_dir):
                logger.info(f'Not exists path for {card_type}')
                continue
            
            outputs[card_type] = []
            dataset = AnoDataset(root=test_dir,
                                 split='test',
                                 icqc2ano=self.icqc2ano,
                                 transforms=self.transforms)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)

            for batch in tqdm(dataloader):
                lst_imgs, lst_paths, lst_targets = batch
                lst_imgs = lst_imgs.to(self.device)
                
                preds, scores = self.infer.predict_batch(lst_imgs, card_type)
                
                for idx in range(len(lst_imgs)):
                    outputs[card_type].append({'path': lst_paths[idx],
                                               'label': lst_targets[idx],
                                               'pred': int(preds[idx][0]),
                                               'score': round(float(scores[idx]), 4),
                                               })
        self.results_test = outputs
        
             
    def visualize_metric(self):
        save_path = os.path.join(self.out_root, 'ROC_Curve.jpg')
        
        for card_type in self.cls_cards:
            target, prediction, scores = [], [], []
            for idx in range(len(self.results_test[card_type])):
                target.append(self.results_test[card_type][idx]['label'])
                prediction.append(self.results_test[card_type][idx]['pred'])
                scores.append(self.results_test[card_type][idx]['score'])
                
            for idx in range(len(self.results_train[card_type])):
                target.append(self.results_train[card_type][idx]['label'])
                prediction.append(self.results_train[card_type][idx]['pred'])
                scores.append(self.results_train[card_type][idx]['score'])
                
            res_metric = visualize_eval(target, prediction, scores, save_path)
            res_metric['threshold_config'] = self.threshs[card_type]
            df = pd.DataFrame([res_metric])
            logger.info(f'Metrics of {card_type}')
            print(df)
            
    def visualize_chart(self):
        for card_type in self.cls_cards:
            name_image = f"{card_type}.jpg"
            save_path = os.path.join(self.out_root, name_image)
            train_score = [self.results_train[card_type][idx]['score'] for idx in range(len(self.results_train[card_type]))]
            test_score = []
            for idx in range(len(self.results_test[card_type])):
                if self.results_test[card_type][idx]['label'] == AnomalyID['abnormal'].value:
                    test_score.append(self.results_test[card_type][idx]['score'])
                else:
                    train_score.append(self.results_test[card_type][idx]['score'])
            lst_score = [train_score, test_score]
            
            draw_chart(scores=lst_score,
                       save_path=save_path,
                       name_images=name_image) 
            
    def pipeline(self, test_root, num_samples, chart, metric):
        if test_root is not None:
            self.inference_test(test_root=test_root)
            
        if chart:
            self.visualize_chart()
            
        if metric:
            self.visualize_metric()
        
        if num_samples > 0:
            self.check_gpu(num_samples)
            
            
def main(config, test_root, num_samples, chart, metric):
    cfg = Config.load_yaml(config)
    vis = VisualAno(cfg)
    
    vis.pipeline(test_root, num_samples, chart, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str)
    parser.add_argument('--test_root', default=None, type=str)
    parser.add_argument('--num_samples', default=0, type=int)
    parser.add_argument('--chart', '-c', default=False, action='store_true')
    parser.add_argument('--metric', '-m', default=False, action='store_true')
    args = parser.parse_args()

    main(config=args.config,
         test_root=args.test_root,
         num_samples=args.num_samples,
         chart=args.chart,
         metric=args.metric)
