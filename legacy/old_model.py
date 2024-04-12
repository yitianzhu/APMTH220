import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.utils import degree, sort_edge_index, to_dense_batch
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv


class SubgraphMamba(nn.Module):
    def __init__(self, hidden_dim, mamba_dim, num_classes, embeddings, edge_tensor, freeze_mamba=False):
        super(SubgraphMamba, self).__init__()
        self.edge_tensor = edge_tensor
        self.embeddings = embeddings

        self.conv1 = GCNConv(embeddings.shape[1], hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mamba_dim-1)
        self.mamba1=Mamba(d_model=mamba_dim, d_state=16, d_conv=4, expand=2)
        # self.mamba2=Mamba(d_model=mamba_dim, d_state=16, d_conv=4, expand=2)
        # self.mamba3=Mamba(d_model=mamba_dim, d_state=16, d_conv=4, expand=2)
        self.fc2 = nn.Linear(mamba_dim + mamba_dim - 1, num_classes)

        self._initialize_weights() 
        
        if freeze_mamba:
            self.freeze_mamba()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, Mamba):
                for name, param in m.named_parameters():
                    param.requires_grad = True
                    if 'weight' in name and 'conv' not in name:  # Exclude conv layers
                        nn.init.kaiming_normal_(param)
    def freeze_mamba(self):
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
        emb = self.conv1(emb, self.edge_tensor, None)
        emb = self.fc1(emb)

        # Preprocessing for subgraph. sequence is a mask 
        subg = emb[sequence]
        last_entry=inclusion.unsqueeze(-1)
        subg = torch.cat((subg, last_entry), dim=-1)
        # Mamba layers
        subg = self.mamba1(subg)
        # subg = self.mamba2(subg)
        
        # aggregation of mpnn embeddings and mamba embeddings via concatenation 
        emb = emb[sequence]
        emb = torch.cat((subg, emb), dim=-1)
        mask=inclusion.unsqueeze(-1)
        emb = emb * mask
        emb = torch.sum(emb, dim=1) / (torch.sum(mask, dim=1)+1e-5)

        # Linear layer for final class, assume 0-1 classification
        emb = self.fc2(emb)
        # No need to apply sigmoid when using CrossEntropyLoss as it is applied internally
        return emb
