import torch
import torch.nn as nn
import torch_geometric.nn as pygnn 
import torch.nn.functional as F

from einops import rearrange 
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.utils import degree, sort_edge_index, to_dense_batch
from torch_geometric.nn import GCNConv

from mamba_ssm import Mamba

from gatedgcn_layer import GatedGCNLayer
from utils import Batch 

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

class GraphMambaLayer(nn.Module):
    def __init__(self, dim_h, dropout=0, attn_dropout=0.2, layer_norm=False, batch_norm=False):
        super(GraphMambaLayer, self).__init__()
        self.dim_h = dim_h
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm 

        # GCN Layer
        if self.layer_norm and self.batch_norm:
            raise ValueError("layer_norm and batch_norm should not be both enabled at the same time.")
        
        self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=False)
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        
        # Attention Layer 
        self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                )

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.layer_norm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, graph_embeddings, graph_edges, sequence):
        # for residual connections 
        h_out_list = []
        h_in1 = graph_embeddings[sequence]

        # Local GNN 
        local_out = self.local_model(Batch(graph_embeddings, graph_edges))
        h_local = local_out.x
        if self.layer_norm:
            h_local = self.norm1_local(h_local)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out_list.append(h_local[sequence])

        # Global mamba layer
        h_attn = self.self_attn(graph_embeddings[sequence])
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn)
        if self.batch_norm:
            h_attn = h_attn.transpose(-1,-2)
            h_attn = self.norm1_attn(h_attn)
            h_attn = h_attn.transpose(-1,-2)
        h_out_list.append(h_attn)

        # Aggregate and feed foward 
        h = sum(h_out_list)
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h)
        if self.batch_norm:
            h = h.transpose(-1,-2)
            h = self.norm2(h)
            h = h.transpose(-1,-2)

        # put h back into the embeddings 
        emb = graph_embeddings.clone() 
        emb[sequence] *= 0.5
        emb[sequence] += h * 0.5
        return emb  
    
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

class GraphMamba(nn.Module):
    def __init__(self, num_classes, embeddings, edge_tensor, hidden_dim=None,
                 n_layers=2, dropout=0.0, attn_dropout=0.2, layer_norm=False, batch_norm=False):
        super(GraphMamba, self).__init__()

        self.num_classes = num_classes
        self.embeddings = embeddings
        self.edge_tensor = edge_tensor

        if hidden_dim==None: 
            hidden_dim = embeddings.size(1)
            self.initial_projection=False 
        else: 
            self.fc1 = nn.Linear(embeddings.size(1), hidden_dim)
            self.initial_projection = True
        self.attn_layers=nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(GraphMambaLayer(hidden_dim, dropout=dropout, attn_dropout=attn_dropout, layer_norm=layer_norm, batch_norm=batch_norm))
        
        # Downstream MLP 
        self.downstream_layers = nn.ModuleList()
        self.downstream_layers.append(nn.Linear(hidden_dim, 2*hidden_dim))
        self.downstream_layers.append(nn.Linear(2*hidden_dim, hidden_dim))
        self.downstream_layers.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, sequence, inclusion):
        if self.initial_projection: 
            emb = self.fc1(self.embeddings)
        else: 
            emb = self.embeddings

        # use Graph Mamba Layer blocks 
        for layer in self.attn_layers: 
            emb = layer(emb, self.edge_tensor, sequence)

        return self.downstream_mlp(emb, sequence, inclusion)
    
    def downstream_mlp(self, emb, sequence, inclusion):
        # Final projection to classes 
        emb = emb[sequence]
        mask=inclusion.unsqueeze(-1)
        emb = emb * mask
        emb = torch.sum(emb, dim=1) / (torch.sum(mask, dim=1)+1e-5)

        for layer in self.downstream_layers:
            emb = layer(emb)
            emb = F.relu(emb)
        return emb 
class SubgraphMamba(nn.Module):
    def __init__(self, hidden_dim, mamba_dim, num_classes, embeddings, edge_tensor, 
    n_layers = 2, dropout=0.2, zero_one_label=False, graph_conv=False, prenorm=True, 
    aggregation = 'concat', freeze_mamba=False):
        super(SubgraphMamba, self).__init__()
        self.edge_tensor = edge_tensor
        self.embeddings = embeddings
        self.graph_conv = graph_conv
        self.zero_one_label = zero_one_label
        self.prenorm = prenorm 
        self.concat_emb = aggregation=='concat'
        self.add_emb = aggregation=='add'

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
        emb = emb[sequence]
        if self.zero_one_label: 
            last_entry=inclusion.unsqueeze(-1)
            emb = torch.cat((emb, last_entry), dim=-1)
        subg = emb.clone()
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
        if self.concat_emb: 
            emb = torch.cat((subg, emb), dim=-1)
        elif self.add_emb:
            emb+=subg
        else:
            emb=subg 
        mask=inclusion.unsqueeze(-1)
        emb = emb * mask
        emb = torch.sum(emb, dim=1) / (torch.sum(mask, dim=1)+1e-5)
        # Linear layer for final class
        emb = self.fc2(emb)
        return emb
