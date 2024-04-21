import torch 
from torch.utils.data import DataLoader, Dataset

class SubgraphDataset(Dataset):
  def __init__(self, subgraph_list, subgraph_labels, degrees, neighbor_dict, num_classes=2, multilabel=False, seqlength=3000, hops=1, augment=1):
    '''
    subgraph_list: list of lists of ints
    subgraph_labels: list of ints
    degrees: tensor where degrees[node_id] is degree of that node
    neighbor_dict: dictionary mapping node_id to ids of neighbors
    seqlength: (optional) maximum sequence length. Append -1 to the start of sequences
    hops: (optional) number of hops away from subgraph to sample
    augment: (optional) number of sequences to make for each subgraph. 
    '''
    self.subgraph_list = []
    self.subgraph_labels = []
    for a in range(augment): 
      for subg, label in zip(subgraph_list, subgraph_labels):
        permuted_subgraph = torch.randperm(len(subg)).tolist()
        self.subgraph_list.append([subg[i] for i in permuted_subgraph])
        self.subgraph_labels.append(label)
    self.degrees = degrees
    self.neighbor_dict = neighbor_dict
    self.seqlength=seqlength
    self.hops=hops
    self.num_classes=num_classes
    self.multilabel=multilabel
  def __len__(self):
    return len(self.subgraph_list)
  def __getitem__(self, idx):
    '''
    Returns a 
    sequence: tensor of shape (self.seqlength, 1) of node IDs in the sequence
    inclusion: tensor of shape (self.seqlength, 1) of indicator for if associated node ID is part of subgraph 
    n: number of nodes in the specific subgraph 
    y: label
    '''
    if isinstance(self.subgraph_labels[idx], torch.Tensor):
      y = self.subgraph_labels[idx].clone().detach()
    else:
      y = torch.tensor(self.subgraph_labels[idx], dtype=torch.long)
    sequence = self._get_sequence_ids(idx)
    padding_length = self.seqlength - len(sequence)
    padding_tensor = torch.full((padding_length,), -1)
    sequence = torch.cat((padding_tensor, sequence))
    inclusion = self._get_inclusion(idx, sequence)
    if self.multilabel:
      y_vec = torch.zeros(self.num_classes, dtype=torch.float)
      y_vec[y]=1
      return sequence, inclusion, len(self.subgraph_list[idx]), y_vec
    return sequence, inclusion, len(self.subgraph_list[idx]), y

    # subgraph_idx = torch.zeros(self.num_graph_nodes)
    # subgraph_idx[self.subgraph_list[idx]] = 1
    # return sequence, inclusion, subgraph_idx, y
  def _sort_by_degree(self, node_ids):
    '''
    node_ids - tensor of the node ids of nodes we want to sort by degree
    largest to smallest bc then we can reverse the tensor at the very last step
    '''
    shuffled_indices = torch.randperm(len(node_ids))
    shuffled_node_ids = node_ids[shuffled_indices]
    degrees = self.degrees[shuffled_node_ids]
    return shuffled_node_ids[torch.argsort(degrees, descending=True)]

  def _get_neighbor_ids(self, sequence):
    '''
    sequence - tensor. this is the previous sequence

    Let v be a node k-1 hops away from the subgraph.
    Then the neighbors of v are sorted by degree but still kept together in the sequence.
    '''
    neighbor_lists=[]
    for s in sequence:
      if s.item() in self.neighbor_dict.keys() and s.item() not in self.current_explored:
        self.current_explored.add(s.item())
        neighbors = torch.IntTensor(self.neighbor_dict[s.item()])
        neighbor_lists.append(self._sort_by_degree(neighbors))
        self.current_seq_len+=len(neighbors)
      if self.current_seq_len >= self.seqlength:
        break
    return torch.cat(neighbor_lists)
  def _get_inclusion(self, idx, sequence):
    '''
    idx: int, index for ith subgraph in the dataset
    sequence: tensor, 1d, with node IDs
    returns: tensor with 0 and 1, same shape as sequence.

    the returned tensor has 1 if the node ID in sequence at that position is included in subgraph.
    '''
    subgraph = torch.IntTensor(self.subgraph_list[idx])
    inclusion = (sequence.unsqueeze(1) == subgraph).any(dim=1).to(torch.float)
    return inclusion
  def _get_sequence_ids(self, idx):
    '''
    idx: int, index ith subgraph in dataset
    returns 1-d tensor of length at most self.seqlength

    Generate a sequence of node IDs associated with a subgraph.
    Nodes further away from the subgraph appear earlier in the sequence.
    Nodes with the same distance from the subgraph are grouped by path to subgraph.
    Nodes within same group are sorted by degree.
    '''
    sequences = [self._sort_by_degree(torch.IntTensor(self.subgraph_list[idx]))]
    self.current_seq_len = len(sequences[0])
    self.current_explored=set()
    for i in range(self.hops):
      if self.current_seq_len < self.seqlength:
        neighbor_ids = self._get_neighbor_ids(sequences[i])
        sequences.append(neighbor_ids[~torch.isin(neighbor_ids, sequences[i])])
        self.current_seq_len=sum(len(sequences[j]) for j in range(len(sequences)))
    return torch.cat(sequences).flip(dims=[0])[-self.seqlength:]
