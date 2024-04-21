import torch 
class Batch:
    def __init__(self, x, edges):
        self.x = x
        self.edge_index = edges
        self.edge_attr = torch.zeros((self.edge_index.size(1), self.x.size(1)), device=self.x.device) 
        
def read_subgraphs(sub_f, split = True):
    '''
    Read subgraphs from file

    Args
       - sub_f (str): filename where subgraphs are stored

    Return for each train, val, test split:
       - sub_G (list): list of nodes belonging to each subgraph
       - sub_G_label (list): labels for each subgraph
    '''

    # Enumerate/track labels
    label_idx = 0
    labels = {}


    # Train/Val/Test subgraphs
    train_sub_G = []
    val_sub_G = []
    test_sub_G = []

    # Train/Val/Test subgraph labels
    train_sub_G_label = []
    val_sub_G_label = []
    test_sub_G_label = []

    # Train/Val/Test masks
    train_mask = []
    val_mask = []
    test_mask = []

    multilabel = False

    # Parse data
    with open(sub_f) as fin:
        subgraph_idx = 0
        for line in fin:
            nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
            if len(nodes) != 0:
                if len(nodes) == 1: print(nodes)
                l = line.split("\t")[1].split("-")
                if len(l) > 1: 
                    multilabel = True
                for lab in l:
                    if lab not in labels.keys():
                        labels[lab] = label_idx
                        label_idx += 1
                if line.split("\t")[2].strip() == "train":
                    train_sub_G.append(nodes)
                    train_sub_G_label.append([labels[lab] for lab in l])
                    train_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "val":
                    val_sub_G.append(nodes)
                    val_sub_G_label.append([labels[lab] for lab in l])
                    val_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "test":
                    test_sub_G.append(nodes)
                    test_sub_G_label.append([labels[lab] for lab in l])
                    test_mask.append(subgraph_idx)
                subgraph_idx += 1
    if not multilabel:
        train_sub_G_label = torch.tensor(train_sub_G_label).long().squeeze()
        val_sub_G_label = torch.tensor(val_sub_G_label).long().squeeze()
        test_sub_G_label = torch.tensor(test_sub_G_label).long().squeeze()
    if multilabel: 
        print("WARNING: Multilabel classification")
    print(labels)
    if len(val_mask) < len(test_mask):
        return train_sub_G, train_sub_G_label, test_sub_G, test_sub_G_label, val_sub_G, val_sub_G_label
    return train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label