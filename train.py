import os
import time
import argparse
from loguru import logger
from tqdm import tqdm
import pandas as pd
# import mlflow

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from coresets.kcenter import KCenterGreedy
from model import PatchCore
from datasets import PatchCoreDataset
from config import Config
from tool import verify_device, get_img_transform, optimal_threshold, visualize_eval, draw_histogram
from const import AnomalyID


class Trainer:
    """ Create checkpoint & analyze performance
    
    Args:
        cfg (dict): config file from config.yml 
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = verify_device(cfg['device'])

        self.cls_cards = cfg['cls_cards']
        self.ckpt = cfg['ckpt']

        self.data_root = cfg['data_root']
        self.test_root = cfg['test_root']

        self.arch = cfg['hparams']['backbone']['arch']
        self.run = cfg['run']
        self.run_name = f"run{cfg['run']}__{self.arch}"

        self.hparams = cfg['hparams']
        model = PatchCore(cfg['hparams'], training=True)
        self.model = model.to(self.device)

        self.transforms = get_img_transform(cfg['hparams']['img_size'])
        self.hp_core = cfg['hparams']['coreset']
        self.batch_size = cfg['hparams']['batch_size']

        self.embed_path = cfg['embedding_path']
        self.out_root = os.path.join(cfg['out_root'], f"run{cfg['run']}")
        os.makedirs(self.out_root, exist_ok=True)

        self.icqc2ano = cfg['icqc2ano']
        self.data_train = self.get_loader(split='train')
        self.data_test = self.get_loader(split='test')

        self.threshs = {}
        self.coresets = {}

    def get_loader(self, split):
        """ Get dataloader for train set & test set with all cards
        
        Args:
            split (str): recognize between train set and test set
            
        Returns:
           dataloader (dict(DataLoader)): dict contain all dataloaders of all cards
        """
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
            dataset = PatchCoreDataset(root=data_dir,
                                       split=split,
                                       icqc2ano=self.icqc2ano,
                                       transforms=self.transforms)

            dataloader[card_type] = DataLoader(dataset,
                                               batch_size=self.batch_size)
            data_info[card_type] = len(dataloader[card_type].dataset)
        print(
            pd.DataFrame({'samples': data_info.values()},
                         index=data_info.keys()))
        return dataloader

    def embedding_coreset(self):
        """ Create embedding coreset and save into self.coresets
        """
        for card_type in self.cls_cards:
            logger.info(f'Embedding {card_type}')

            for idx, batch in enumerate(tqdm(self.data_train[card_type])):
                images, _, _ = batch
                images = images.to(self.device)
                embedding = self.model(images, embedding_coreset=None)
                if idx == 0:
                    embedding_vectors = embedding
                else:
                    embedding_vectors = torch.cat(
                        (embedding_vectors, embedding), 0)

            self.coresets[card_type] = self.subsample_embedding(
                embedding_vectors)
            logger.info('-' * 20)

    def subsample_embedding(self, embedding):
        """ Create coreset with feature extracted from the backbone
        
        Args:
            embedding (Tensor): feature extracted from backbone configuration 
             
        Returns:
            coreset (Tensor): coreset with configuration from config file
        """
        sampler = KCenterGreedy(embedding=embedding,
                                sampling_ratio=self.hp_core['ratio'],
                                eps=self.hp_core['eps'],
                                seed=self.hp_core['seed'])
        coreset = sampler.sample_coreset()
        return coreset

    def fit(self):
        """ Pipeline with task steps:
            1: Create embedding coreset
            2: Find optimal threshold and save metrics for each type of card
            3: Compute hardware used for singer data
            4: Save checkpoint 
        """
        self.embedding_coreset()

        self.model.training = False
        metric = {}

        logger.info(f'Analyze train & test')
        # with mlflow.start_run(run_name=self.run_name) as run:
        #     mlflow.log_param("Hyper-params", self.hparams)

        for card_type in self.cls_cards:
            logger.info(f"Class card: {card_type}")
            result_train, mem_train, time_train = self.inference_one_card(
                self.data_train[card_type], card_type)
            result_test, mem_test, time_test = self.inference_one_card(
                self.data_test[card_type], card_type)

            all_results = result_train + result_test
            targets, scores = [], []
            for idx in range(len(all_results)):
                targets.append(all_results[idx]['label'])
                scores.append(all_results[idx]['score'])

            _, _, threshold = optimal_threshold(targets, scores)
            self.threshs[card_type] = threshold

            result_train = self.append_prediction(result_train, threshold)
            result_test = self.append_prediction(result_test, threshold)

            df = self.compute_metric(result_train, result_test, threshold)
            df['GPU Memory'] = (mem_train + mem_test) / 2
            df['Time'] = (time_train + time_test) / 2
            logger.info(f'Metrics of {card_type}')
            metric[card_type] = df
            print(df)
            # mlflow.log_metric(f"Metric {card_type}", df)

            name_image = f"{card_type}.jpg"
            fig = self.save_histogram(result_train, result_test,
                                        name_image)
            # mlflow.log_figure(f"Histogram {card_type}", fig)

            result_path = os.path.join(self.out_root,
                                        f"{self.run_name}__{card_type}.pt")
            save_results = {'train': result_train, 'test': result_test}
            torch.save(save_results, result_path)

        metric_path = os.path.join(self.out_root,
                                    f"{self.run_name}__metric.pt")
        torch.save(metric, metric_path)

        ckpt_path = os.path.join(self.embed_path, f"{self.run_name}.pt")
        if self.ckpt is not None:
            info_ver = torch.load(self.ckpt)
        else:
            info_ver = {}
        for card_type in self.cls_cards:
            info_ver[card_type] = {
                'run': self.run,
                'hparams': self.hparams,
                'threshold': self.threshs[card_type],
                'embedding_coreset': self.coresets[card_type],
            }
        torch.save(info_ver, ckpt_path)
        # mlflow.pytorch.log_model(info_ver, "Model PatchCore")

    def inference_one_card(self, dataloader,
                           card_type):
        """ Compute scores, GPU memory and time inference for one dataloader
        
        Args:
            dataloader (DataLoader): dataloader for train or test in one card type
            card_type (str): name of card type
            
        Returns:
            (output, memory, time) tuple(list, Tensor, float): the results after computed
        """
        output, time_infer, gpu_memories = [], [], []
        num_samples = len(dataloader.dataset)
        for batch in tqdm(dataloader):
            start_time = time.perf_counter()

            lst_imgs, lst_paths, lst_targets = batch
            lst_imgs = lst_imgs.to(self.device)
            _, scores = self.model(lst_imgs, self.coresets[card_type])

            time_infer.append(time.perf_counter() - start_time)
            gpu_memories.append(torch.cuda.memory_allocated())

            for idx in range(len(lst_imgs)):
                output.append({
                    'path': lst_paths[idx],
                    'label': lst_targets[idx],
                    'score': round(float(scores[idx]), 4),
                })

        memory = sum(gpu_memories) / (1024**3 * num_samples)
        time_per = sum(time_infer) * 100 / num_samples
        return output, memory, time_per

    def append_prediction(self, results,
                          threshold):
        """ Append predict with optimal threshold into results 

        Args:
            results (list(dict)): list contian results for train or test in one card type
            threshold (float): optimal threshold for one card type

        Returns:
            results (list(dict)): results added prediction
        """
        for idx in range(len(results)):
            score = results[idx]['score']
            if score >= threshold:
                pred = AnomalyID['abnormal'].value
            else:
                pred = AnomalyID['normal'].value
            results[idx]['pred'] = pred
        return results

    def compute_metric(self, result_train, result_test,
                       threshold):
        """ Compute and return precision, recall and ROC

        Args:
            result_train (list(dict)): list contian results for train in one card type
            result_test (list(dict)): list contian results for test in one card type
            threshold (float): optimal threshold for one card type

        Returns:
            pd.DataFrame: contains the computed values of precision, recall, ROC and optimal threshold 
        """
        save_path = os.path.join(self.out_root, 'ROC_Curve.jpg')
        all_results = result_train + result_test

        target, prediction, scores = [], [], []
        for result in all_results:
            target.append(result['label'])
            prediction.append(result['pred'])
            scores.append(result['score'])

        res_metric = visualize_eval(target, prediction, threshold, save_path)
        return pd.DataFrame([res_metric])

    def save_histogram(self, result_train, result_test,
                       name_image):
        """ Draw histograms with scores of train set and test set

        Args:
            result_train list(dict): list contain results for train in one card type
            result_test list(dict): list contain results for test in one card type
            name_image (str): name of histograms
        """
        train_score = [
            result_train[idx]['score'] for idx in range(len(result_train))
        ]
        test_score = []
        for idx in range(len(result_test)):
            if result_test[idx]['label'] == AnomalyID['abnormal'].value:
                test_score.append(result_test[idx]['score'])
            else:
                train_score.append(result_test[idx]['score'])
        lst_score = [train_score, test_score]

        fig = draw_histogram(scores=lst_score,
                             save_path=self.out_root,
                             name_images=name_image)
        return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    cfg = Config.load_yaml(args.config)
    train = Trainer(cfg)

    logger.info(f"Data root: {cfg['data_root']}")
    train.fit()
    logger.info(f"Successfully & save path: {cfg['embedding_path']}")


if __name__ == '__main__':
    main()
