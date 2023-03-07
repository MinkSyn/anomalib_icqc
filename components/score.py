import torch
import torch.nn.functional as F
from torch import nn, Tensor


class AnomalyScores(nn.Module):
    """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.
        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.
        Args:
            cfg (dict): Input tensor
        """

    def __init__(self, cfg):
        self.method_dis = cfg['distance']
        self.num_neighbors = cfg['num_neighbors']

    def calculate_distance(self, input_embedding, embedding_coreset):
        if self.method_dis == 'euclidean':
            distances = torch.cdist(input_embedding, embedding_coreset, p=2.0)
        return distances

    def nearest_neighbors(self, input_embedding, embedding_coreset,
                          n_neighbors):
        """Nearest Neighbours using brute force method and euclidean norm.
        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at
        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = self.calculate_distance(input_embedding, embedding_coreset)
        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors,
                                                     largest=False,
                                                     dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, embedding_coreset, patch_scores, locations,
                              embedding):
        """Compute Image-Level Anomaly Score.
        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """
        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        max_patches = torch.argmax(patch_scores, dim=1)
        max_patches_features = embedding.reshape(batch_size, num_patches,
                                                 -1)[torch.arange(batch_size),
                                                     max_patches]

        score = patch_scores[torch.arange(batch_size), max_patches]
        nn_index = locations[torch.arange(batch_size), max_patches]

        nn_sample = embedding_coreset[nn_index, :]
        _, support_samples = self.nearest_neighbors(
            nn_sample, embedding_coreset, n_neighbors=self.num_neighbors)

        distances = self.calculate_distance(max_patches_features.unsqueeze(1),
                                            embedding_coreset[support_samples])
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]

        score = weights * score
        return score

    def forward(self, embedding: Tensor, batch_size: int,
                embedding_coreset: Tensor) -> list:
        """ Return scores of test embedding

        Args:
            embedding (Tensor): feature from CNN generated to embedding
            batch_size (int): number of embedding
            embedding_coreset (Tensor): _description_

        Returns:
            scores (list): values computed of embedding 
        """
        patch_scores, locations = self.nearest_neighbors(
            input_embedding=embedding,
            embedding_coreset=embedding_coreset,
            n_neighbors=1)

        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))

        scores = self.compute_anomaly_score(embedding_coreset, patch_scores,
                                            locations, embedding)
        return scores