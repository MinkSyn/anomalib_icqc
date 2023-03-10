import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from patchcore.backbone import TimmFeatureExtractor
from patchcore.components.anomaly_map import AnomalyMapGenerator
from patchcore.components.pre_process import Tiler
from patchcore.components.dynamic_module import DynamicBufferModule
from patchcore.sampling_methods.kcenter import KCenterGreedy


class PatchcoreModel(DynamicBufferModule, nn.Module):
    def __init__(self,
                 input_size,
                 layers,
                 method_dis,
                 backbone = "wide_resnet50_2",
                 pre_trained = True,
                 training = False,
                 eps = 0.9,
                 seed = 101,
                 sampling_ratio = 0.1,
                 num_neighbors = 9.):
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        
        self.training = training
        
        self.method_dis = method_dis
        self.num_neighbors = num_neighbors
        
        self.sampling_ratio = sampling_ratio
        self.eps = eps
        self.seed = seed

        self.feature_extractor = TimmFeatureExtractor(backbone=self.backbone, 
                                                  pre_trained=pre_trained, 
                                                  layers=self.layers)
        
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", Tensor())
        self.memory_bank = None

    def forward(self, input_tensor, embedding_coreset):
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.
        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.
        Args:
            input_tensor (Tensor): Input tensor
        Returns:
            Tensor | tuple[Tensor, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            # apply nearest neighbor search
            patch_scores, locations = self.nearest_neighbors(input_embedding=embedding, 
                                                             embedding_coreset=embedding_coreset,
                                                             n_neighbors=1)
            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
            # compute anomaly score
            anomaly_score = self.compute_anomaly_score(embedding_coreset, patch_scores, locations, embedding)
            # reshape to w, h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)

            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features):
        """Generate embedding from hierarchical feature map.
        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:
        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding):
        """Reshape Embedding.
        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]
        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.
        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding
    
    def subsample_embedding(self, embedding: Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.
        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, 
                                sampling_ratio=self.sampling_ratio,
                                eps=self.eps,
                                seed=self.seed)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    
    def calculate_distance(self, input_embedding, embedding_coreset):
        if self.method_dis == 'euclidean':
            distances = torch.cdist(input_embedding, embedding_coreset, p=2.0)
            
        return distances

    def nearest_neighbors(self, input_embedding, embedding_coreset, n_neighbors):
        """Nearest Neighbours using brute force method and euclidean norm.
        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at
        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = self.calculate_distance(input_embedding, embedding_coreset)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, embedding_coreset, patch_scores, locations, embedding):
        """Compute Image-Level Anomaly Score.
        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = embedding_coreset[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        _, support_samples = self.nearest_neighbors(nn_sample, embedding_coreset, n_neighbors=self.num_neighbors)
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.calculate_distance(max_patches_features.unsqueeze(1), embedding_coreset[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score
    
    
def build_model(cfg, training):
    model = PatchcoreModel(input_size=cfg['feature']['img_size'],
                           layers=cfg['feature']['layer'],
                           backbone=cfg['feature']['arch'],
                           pre_trained=cfg['feature']['pretrained'],
                           training=training,
                           eps=cfg['para']['eps'],
                           seed=cfg['para']['seed'],
                           sampling_ratio=cfg['hyper']['ratio'],
                           num_neighbors=cfg['hyper']['num_neighbors'],
                           method_dis=cfg['hyper']['distance'],)

    return model

if __name__ == '__main__':
    model = build_model()
