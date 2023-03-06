from __future__ import annotations

import logging
import warnings

import timm
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class TimmFeatureExtractor(nn.Module):
    """Extract features from a CNN.
    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.
    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import TimmFeatureExtractor
        >>> model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)
        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self,
                 backbone,
                 layers,
                 pre_trained=True,
                 requires_grad=False):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        self.feature_extractor = self._get_backbone(backbone,
                                                    pretrained=pre_trained)
        self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}
        
    def _get_backbone(self, backbone, pretrained):
        if isinstance(pretrained, bool):
            feature = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         features_only=True,
                                         exportable=True,
                                         out_indices=self.idx,
                                         )
        elif isinstance(pretrained, str):
            feature = timm.create_model(backbone,
                                         pretrained=False,
                                         features_only=True,
                                         exportable=True,
                                         out_indices=self.idx,
                                         )
            model_state = torch.load(pretrained, map_location="cpu")
            # print(model_state.keys())
            # exit()
            print(f"Loading weight {pretrained}")
            try:
                feature.load_state_dict(model_state)
            except:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state.items():
                    name = k.replace('module.', '', 1)  # remove 'module.' of dataparallel
                    name = k.replace('model.', '', 1)
                    new_state_dict[name] = v
                feature.load_state_dict(new_state_dict)
        return feature   

    def _map_layer_to_idx(self, offset=3):
        """Maps set of layer names to indices of model.
        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need
        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        features = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        for i in self.layers:
            try:
                idx.append(
                    list(dict(features.named_children()).keys()).index(i) -
                    offset)
            except ValueError:
                warnings.warn(f"Layer {i} not found in model {self.backbone}")
                # Remove unfound key from layer dict
                self.layers.remove(i)

        return idx

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.
        Args:
            inputs (Tensor): Input tensor
        Returns:
            Feature map extracted from the CNN
        """
        if self.requires_grad:
            features = dict(zip(self.layers, self.feature_extractor(inputs)))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(
                    zip(self.layers, self.feature_extractor(inputs)))
        return features
