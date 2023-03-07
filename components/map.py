from __future__ import annotations

import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn

from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d


def compute_kernel_size(sigma_val):
    """Compute kernel size from sigma value.
    Args:
        sigma_val (float): Sigma value.
    Returns:
        int: Kernel size.
    """
    return 2 * int(4.0 * sigma_val + 0.5) + 1


class GaussianBlur2d(nn.Module):
    """Compute GaussianBlur in 2d.
    Makes use of kornia functions, but most notably the kernel is not computed
    during the forward pass, and does not depend on the input size. As a caveat,
    the number of channels that are expected have to be provided during initialization.
    """

    def __init__(self,
                 sigma,
                 channels=1,
                 kernel_size=None,
                 normalize=True,
                 border_type="reflect",
                 padding="same"):
        """Initialize model, setup kernel etc..
        Args:
            sigma (float | tuple[float, float]): standard deviation to use for constructing the Gaussian kernel.
            channels (int): channels of the input. Defaults to 1.
            kernel_size (int | tuple[int, int] | None): size of the Gaussian kernel to use. Defaults to None.
            normalize (bool, optional): Whether to normalize the kernel or not (i.e. all elements sum to 1).
                Defaults to True.
            border_type (str, optional): Border type to use for padding of the input. Defaults to "reflect".
            padding (str, optional): Type of padding to apply. Defaults to "same".
        """
        super().__init__()
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.channels = channels

        if kernel_size is None:
            kernel_size = (compute_kernel_size(sigma[0]),
                           compute_kernel_size(sigma[1]))
        else:
            kernel_size = kernel_size if isinstance(
                kernel_size, tuple) else (kernel_size, kernel_size)

        self.kernel: Tensor
        self.register_buffer(
            "kernel",
            get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)
        self.kernel.unsqueeze_(0).unsqueeze_(0)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor):
        """Blur the input with the computed Gaussian.
        Args:
            input_tensor (Tensor): Input tensor to be blurred.
        Returns:
            Tensor: Blurred output tensor.
        """
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor,
                                 self.padding_shape,
                                 mode=self.border_type)

        output = F.conv2d(input_tensor,
                          self.kernel,
                          groups=self.channels,
                          padding=0,
                          stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1,
                              width - self.width + 1)
        return out


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, input_size, sigma=4):
        super().__init__()
        self.input_size = input_size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size),
                                   sigma=(sigma, sigma),
                                   channels=1)

    def compute_anomaly_map(self, patch_scores):
        """Pixel Level Anomaly Heatmap.
        Args:
            patch_scores (Tensor): Patch-level anomaly scores
        Returns:
            Tensor: Map of the pixel-level anomaly scores
        """
        anomaly_map = F.interpolate(patch_scores,
                                    size=(self.input_size[0],
                                          self.input_size[1]))
        anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    def forward(self, patch_scores):
        """Returns anomaly_map and anomaly_score.
        Args:
            patch_scores (Tensor): Patch-level anomaly scores
        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map = anomaly_map_generator(patch_scores=patch_scores)
        Returns:
            Tensor: anomaly_map
        """
        maps = self.compute_anomaly_map(patch_scores)
        return maps
