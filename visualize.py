import os
import time
import argparse
from loguru import logger

import pandas as pd
import torch

from config import Config
from tool import verify_device, visualize_eval, draw_chart
from const import AnomalyID
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
        
        self.embedded_path = os.path.join(cfg['embedding_path'], f"{self.run_name}.pt")
        ckpt = self.get_config(self.embedded_path)
        self.threshs = ckpt['threshold']
        self.cls_cards = self.threshs.keys()
        
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
        infer = AnoInference(self.device, self.embedded_path)
        for card_type in self.cls_cards:
            if num_samples > len(self.results_train[card_type]):
                num_samples = len(self.results_train[card_type])
                
            time_infer, gpu_memory = [], [],
            logger.info(f"Check gpu for {card_type}")
            for idx in range(num_samples):
                img_path = self.results_train[card_type][idx]['path']
                start_time = time.perf_counter()
                _ = infer.predict_single(img_path, card_type)
                time_infer.append(time.perf_counter() - start_time)
                gpu_memory.append(torch.cuda.memory_allocated())
            logger.info(f"GPU memory: {sum(gpu_memory) / (1024**3 * num_samples)}")
            logger.info(f"Time inference: {sum(time_infer) / num_samples}")
            logger.info("-"*50)
             
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
            
            
def main(config, num_samples, chart, metric):
    cfg = Config.load_yaml(config)
    vis = VisualAno(cfg)
    
    if chart:
        vis.visualize_chart()
        
    if metric:
        vis.visualize_metric()
    
    if num_samples > 0:
        vis.check_gpu(num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--num_samples', default=0, type=int)
    parser.add_argument('--chart', '-c', default=False, action='store_true')
    parser.add_argument('--metric', '-m', default=False, action='store_true')
    args = parser.parse_args()

    main(config=args.config,
         num_samples=args.num_samples,
         chart=args.chart,
         metric=args.metric)
