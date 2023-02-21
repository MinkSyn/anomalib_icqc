import os
import json
import argparse
from loguru import logger
from collections import Counter
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import DataLoader

from patchcore.model import build_model
from patchcore.sampling_methods.kcenter import KCenterGreedy
from config import Config
from tool import verify_device, draw_chart, binary_classify, infer_transform
from const import CardID, nameAno, IMG_SIZE


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

    def predict(self, test_name=None, visual=False, coreset=False):
        os.makedirs(os.path.join(self.out_root, test_name), exist_ok=True)
        result_path = os.path.join(self.out_root, test_name, f"{self.run_name}.json")
        
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            logger.info(f'Successfully loaded the file: f"{test_name}/{self.run_name}.json"')
        else:
            results = {}
            logger.info(f"Test name: {test_name}")
            for cls_name in self.cls_card:
                if coreset:
                    test_dir = os.path.join(self.data_root, cls_name, 'good')
                else:
                    test_dir = os.path.join(self.test_root, cls_name, 'bad', test_name)
                if not os.path.exists(test_dir):
                    logger.info(f"Not search path for [{cls_name}]")
                    continue
                
                embedding_coreset = self.load_coreset(cls_name=cls_name)
                logger.info(f'Inference: [{cls_name}]')
                
                outputs = self.predict_card(embedding_coreset, test_dir, cls_name)
                results[cls_name] = outputs
                logger.info('-'*20)

            with open(result_path, 'w') as g:
                json.dump(results, g, indent=4)
                                  
        if visual:
            self.visualization(results, test_name)
            
        return results
        
    def predict_card(self, embedding_coreset, test_dir, cls_name):
        name_imgs = sorted(os.listdir(test_dir))
        images = []
        for name in name_imgs:
            img_path = os.path.join(test_dir, name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, IMG_SIZE)
            images.append((image, img_path, name))
            
        dataloader = DataLoader(images, batch_size=self.batch_size)
        
        outputs = []
        for batch in tqdm(dataloader):
            lst_imgs = batch[0].numpy()
            # lst_paths = batch[1]
            # lst_names = batch[2]
            ano_batch = infer_transform(lst_imgs, self.device)
            anomaly_map, anomaly_score = self.model(ano_batch, embedding_coreset)
            pred = binary_classify(anomaly_score, self.thresh[cls_name])
        
            for idx in range(len(lst_imgs)):
                outputs.append({'pred': nameAno[int(pred[idx])],
                                'prob': round(float(anomaly_score[idx]), 4),
                                })
        return outputs
    
    def visualization(self, results_test, test_name):
        results_train = self.predict(test_name='train_embedding', coreset=True)
        save_path = os.path.join(self.out_root, test_name)

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
            
def main(cls_quality='occluded', visual=False):
    config = Config.load_yaml('config.yml')
    infer = Inference(config)
    
    results = infer.predict(test_name=cls_quality,
                            visual=visual)
    # exit()
    for cls_name in results.keys():
        predict = [results[cls_name][idx]['pred'] for idx in range(len(results[cls_name]))]
        view_res = Counter(predict)
        logger.info(f"{cls_name}: {view_res}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_quality', default='')
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--visual', '-v', default=False, action='store_true')
    args = parser.parse_args()

    main(cls_quality=args.cls_quality, 
         visual=args.visual,
         config=args.config)
