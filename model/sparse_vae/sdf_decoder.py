import torch.nn as nn

from model.utils.utils_mlp import MLP


class SDFDecoder(nn.Module):
    """Decode interpolated sparse features into SDF values.

    Args:
        dim_feat (int): dimension of latent conditioned code
        hidden_size (int): hidden size of the decoder network
    """

    def __init__(self, dim_feat=128, hidden_size=256):
        super().__init__()
        self.mlp = MLP(dim_feat, hidden_size, 1, num_layers=5)

    def forward(self, _coordinates, feat_p):
        return self.mlp(feat_p)
