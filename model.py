import torch
import torch.nn.functional as F
from torch import Tensor, nn

from components.feature import TimmFeatureExtractor
from components.map import AnomalyMapGenerator
from components.score import AnomalyScores


class PatchCore(nn.Module):
    """ Use return embedding or return computed scores & maps follow PatchCore model

    Args:
        hparams (dict): hyper-params to build the model  
        training (bool): recognize between feature extraction and compute scores & maps 
    """

    def __init__(self, hparams, training=False):
        super().__init__()
        self.hparams = hparams

        if self.training:
            pre_trained = self.hparams['backbone']['pretrained']
        self.training = training

        layers = self.hparams['backbone']['layer']
        arch = self.hparams['backbone']['arch']
        self.feature_extractor = TimmFeatureExtractor(backbone=arch,
                                                      pre_trained=pre_trained,
                                                      layers=layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)

        self.compute_scores = AnomalyScores(cfg=self.hparams['nn_search'])
        self.map_generator = AnomalyMapGenerator(
            input_size=self.hparams['img_size'])

    def generate_embedding(self, features: dict[str:Tensor]) -> Tensor:
        """ Generate embedding from hierarchical feature map.
        
        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:
        
        Returns:
            Embedding vector
        """
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding,
                                            size=embeddings.shape[-2:],
                                            mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        return embeddings

    def reshape_embedding(self, embedding: Tensor) -> Tensor:
        """ Reshape Embedding.
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

    def forward_feature(self,
                        input_tensor: Tensor) -> tuple(Tensor, int, int, int):
        """ Feature extraction forward a backbone and poolings layer

        Args:
            input_tensor (Tensor): images after pre-processing

        Returns:
            (embedding, batch_size, width, height) tuple(Tensor, int, int, int): values used to compute scores & maps
        """
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {
            layer: self.feature_pooler(feature)
            for layer, feature in features.items()
        }
        embedding = self.generate_embedding(features)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        return embedding, batch_size, width, height

    def forward(self, x: Tensor, card_type: str) -> tuple(list, list):
        """ Pipeline of PatchCore model

        Args:
            x (Tensor): images after pre-processing
            card_type (str): name of card type
            
        Returns:
            scores, maps (tuple(Tensor, Tensor)): results of scores & maps after computed
        """
        x, batch_size, width, height = self.forward_feature(x)
        if self.training:
            return x
        scores, patch_scores = self.compute_scores(x, batch_size,
                                                   self.coresets[card_type])

        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        maps = self.map_generator(patch_scores)

        return scores, maps
