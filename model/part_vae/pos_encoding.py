import torch.nn as nn


def patch_grid_size(sequence_length):
    grid_size = round(sequence_length ** (1 / 3))
    if grid_size ** 3 != sequence_length:
        raise ValueError(
            'Stage 2 decoder sequence length must define a cubic grid: '
            f'{sequence_length}'
        )
    return grid_size


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute embedding for integer 3D patch coordinates."""

    def __init__(self, conf):
        super().__init__()
        self.n_emd = conf['model.pe.n_emb']
        self.dim_emb = conf['model.pe.dim_emb']

        self.emb_all = nn.Embedding(self.n_emd, self.dim_emb)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.emb_all.weight)

    def forward(self, inp):
        """Embed integer code coordinates shaped T x B x 3."""
        inp_vox = inp.clamp(0, self.n_emd - 1)
        return self.emb_all(inp_vox).flatten(-2, -1)
