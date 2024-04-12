import torch
import torch.nn as nn
from einops import rearrange 
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.utils import degree, sort_edge_index, to_dense_batch
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X 

class SubgraphMamba(nn.Module):
    def __init__(self, hidden_dim, mamba_dim, num_classes, embeddings, edge_tensor, 
    n_layers = 2, dropout=0.2, zero_one_label=False, graph_conv=False, prenorm=True, 
    concat_emb=False, freeze_mamba=False):
        super(SubgraphMamba, self).__init__()
        self.edge_tensor = edge_tensor
        self.embeddings = embeddings
        self.graph_conv = graph_conv
        self.zero_one_label = zero_one_label
        self.prenorm = prenorm 
        self.concat_emb = concat_emb

        if self.zero_one_label: 
            pre_mamba_dim = mamba_dim - 1
        else:
            pre_mamba_dim = mamba_dim 

        if graph_conv: 
            self.conv1 = GCNConv(embeddings.shape[1], hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, pre_mamba_dim)
        else: 
            self.fc1 = nn.Linear(embeddings.shape[1], pre_mamba_dim)
        
        self.mamba_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.mamba_layers.append(Mamba(d_model=mamba_dim, d_state=16, d_conv=4, expand=2))
            self.norms.append(RMSNorm(mamba_dim))
            self.dropouts.append(DropoutNd(p=dropout))

        if self.concat_emb: 
            self.fc2 = nn.Linear(mamba_dim+pre_mamba_dim, num_classes)
        else: 
            self.fc2 = nn.Linear(mamba_dim, num_classes)

        self._initialize_weights() 

        if freeze_mamba:
            self._freeze_mamba()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, Mamba):
                for name, param in m.named_parameters():
                    param.requires_grad = True
                    if 'weight' in name and 'conv' not in name:  # Exclude conv layers
                        nn.init.kaiming_normal_(param)
    def _freeze_mamba(self):
        for m in self.modules():
            if isinstance(m, Mamba):
                for name, param in m.named_parameters():
                    param.requires_grad = False
    def print_mamba_gradients(self):
        for m in self.modules():
            if isinstance(m, Mamba):
                for name, param in m.named_parameters():
                    if param.grad is not None:
                        print(f"Gradient for {name}: {param.grad}")
                    else:
                        print(f"No gradient for {name}")
    
    def forward(self, sequence, inclusion, subgraph_size=None):
        # start with linear layer projection
        emb = self.embeddings
        if self.graph_conv: 
            emb = self.conv1(emb, self.edge_tensor, None)
        emb = self.fc1(emb)

        # Preprocessing for subgraph. sequence is a mask 
        subg = emb[sequence]
        if self.zero_one_label: 
            last_entry=inclusion.unsqueeze(-1)
            subg = torch.cat((subg, last_entry), dim=-1)
        # Mamba layers
        for layer, norm, dropout in zip(self.mamba_layers, self.norms, self.dropouts):
            z = subg 
            # prenormalize 
            if self.prenorm:
                z = norm(z)
            # mamba
            z = layer(z)
            # dropout 
            z=dropout(z)
            # residual 
            subg = subg + z 
            # postnormalize 
            if not self.prenorm:
                subg = norm(subg)

        # aggregation of mpnn embeddings and mamba embeddings via concatenation 
        emb = emb[sequence]
        if self.concat_emb: 
            emb = torch.cat((subg, emb), dim=-1)
        mask=inclusion.unsqueeze(-1)
        emb = emb * mask
        emb = torch.sum(emb, dim=1) / (torch.sum(mask, dim=1)+1e-5)

        # Linear layer for final class, assume 0-1 classification
        emb = self.fc2(emb)
        # No need to apply sigmoid when using CrossEntropyLoss as it is applied internally
        return emb
