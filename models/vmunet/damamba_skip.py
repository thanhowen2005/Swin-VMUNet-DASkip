"""Adapt the original DAMamba Block to Swin-VMUNet's BHWC skip tensors."""

from torch import nn


class DAMambaSkipBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        try:
            import DCNv3  # noqa: F401
            import selective_scan_cuda_oflex_rh  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DAMamba skip connections require the original DCNv3 and "
                "selective_scan_cuda_oflex_rh CUDA extensions. Run "
                "'bash install_damamba.sh' from Swin-VMUNet with the Python "
                "environment used for training. All extension sources are "
                "included in models/vmunet/damamba_original."
            ) from exc

        from .damamba_original.DAMamba import Block, DASSM

        self.block = Block(
            dim=dim,
            token_mixer=DASSM,
            head_dim=16,
            mlp_ratio=4,
            drop_path=0.,
            ffnconv=True,
            layerscale=False,
        )

    def forward(self, skip):
        skip = skip.permute(0, 3, 1, 2).contiguous()
        skip = self.block(skip)
        return skip.permute(0, 2, 3, 1).contiguous()
