import os
import glob 
import json 
import datetime
import argparse
import logging 
from time import time 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

from model import SubgraphMamba 
from dataset import SubgraphDataset
from utils import read_subgraphs


# Parse command line arguments for the dataset
parser = argparse.ArgumentParser(description='Subgraph Mamba Learning')
parser.add_argument('--dataset', type=str, required=True, help='Dataset string')
parser.add_argument('--freeze-mamba', action='store_true', help='Freeze Mamba flag')
parser.add_argument('--logfilename', type=str, default=None, help='Save to log file')
args = parser.parse_args()

dataset = args.dataset
freeze_mamba = args.freeze_mamba
logfilename = args.logfilename
if logfilename is None: 
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = f"dumpster/{current_datetime}_{dataset}.log"
dataset_filename = f"data/{dataset}/"

logging.basicConfig(filename=f'what_did_we_learn/{logfilename}', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if freeze_mamba:
    logging.info("Freezing Mamba layers to compare performance")

with open(dataset_filename+'config.json', 'r') as f:
    config = json.load(f)
    logging.info(f"Current date and time:{datetime.datetime.now()}")
    logging.info(config)
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
batch_size = config['batch_size']
epochs = config['epochs']
patience = config['patience']
hidden_dim = config['hidden_dim']
mamba_dim = config['mamba_dim']
num_classes = config['num_classes']
embeddings_filename = config['embeddings_filename']
multilabel = config['multilabel']
seqlength = config['seqlength']
hops = config['hops']
augment = config['augment']


# neighbors of each node in a dict
with open(dataset_filename+"ego_graphs.txt", "r") as file:
    file_contents = file.read()
    ego_graphs = json.loads(file_contents)
neighbor_dict = {int(key): value for key, value in ego_graphs.items()}

# degrees of each node in a tensor
with open(dataset_filename+"degree_sequence.txt", "r") as file:
  file_contents=file.read()
  degree_sequence = json.loads(file_contents)
degrees = torch.zeros(len(degree_sequence))
for key, value in degree_sequence.items():
  degrees[int(key)]=value

# Graph nodes and edges
edge_tensor = torch.load(dataset_filename+'edge_tensor.pth', torch.device('cpu'))
pretrained_node_embeds = torch.load(dataset_filename+embeddings_filename, torch.device('cpu'))
embeddings = torch.cat((pretrained_node_embeds, torch.zeros(1, pretrained_node_embeds.shape[1])), axis=0)

# Read subgraphs
train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label = read_subgraphs(dataset_filename+"subgraphs.pth", split = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = SubgraphDataset(train_sub_G, train_sub_G_label, degrees, neighbor_dict, num_classes, multilabel, seqlength, hops, augment)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = SubgraphDataset(val_sub_G, val_sub_G_label, degrees, neighbor_dict, num_classes, multilabel, seqlength, hops, augment)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

embeddings = embeddings.to(device)
edge_tensor = edge_tensor.to(device)

model = SubgraphMamba(hidden_dim, mamba_dim, num_classes, embeddings, edge_tensor, freeze_mamba=freeze_mamba)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
if multilabel:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
model.to(device)

def train():
    max_acc = 0.00
    patience_cnt = 0
    val_acc_values = []
    best_epoch = 0

    t = time()
    model.train()
    for epoch in range(epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            sequence, inclusion, subgraph_size, y = data
            optimizer.zero_grad()
            sequence = sequence.to(device)
            inclusion = inclusion.to(device)
            subgraph_size = subgraph_size.to(device)
            y=y.to(device)
            out = model(sequence, inclusion, subgraph_size) 

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if torch.isnan(loss):
                logging.info(out)
                logging.info(y)
                return 
            loss_train += loss.item()
            if multilabel:
                pred = torch.sigmoid(out).data > 0.5
            else: 
                pred = out.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
        if multilabel: 
            acc_val, loss_val = compute_test_multilabel(val_loader)
            acc_train = correct / (len(train_loader.dataset)*num_classes)
        else: 
            acc_val, loss_val = compute_test(val_loader)
            acc_train = correct / (len(train_loader.dataset))
        
        information = ('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train),
              'acc_val: {:.4f}'.format(acc_val))
        log_str = ' '.join(information)
        logging.info(log_str)

        val_acc_values.append(acc_val)
        # torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_acc_values[-1] > max_acc:
            max_acc = val_acc_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == patience:
            break

        '''
        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)
        '''

    '''files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    '''
    logging.info('Optimization Finished! Total time elapsed: {:.6f}'.format(time() - t))

    return best_epoch

def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        sequence, inclusion, subgraph_size, y = data
        sequence = sequence.to(device)
        inclusion = inclusion.to(device)
        subgraph_size = subgraph_size.to(device)
        y=y.to(device)
        out = model(sequence, inclusion, subgraph_size)
        pred = out.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        loss_test += criterion(out, y).item()
    return correct / len(loader.dataset), loss_test

def compute_test_multilabel(loader):
    model.eval()
    total_samples = 0
    loss_test = 0.0
    total_correct = 0.0
    for data in loader:
        sequence, inclusion, subgraph_size, y = data
        sequence = sequence.to(device)
        inclusion = inclusion.to(device)
        subgraph_size = subgraph_size.to(device)
        y = y.to(device)
        out = model(sequence, inclusion, subgraph_size)
        loss_test += criterion(out, y.float()).item()  # Ensure y is a float tensor
        pred = torch.sigmoid(out).data > 0.5  # Threshold predictions
        total_correct += pred.eq(y).sum().item()  # Count correct predictions
        total_samples += y.size(0) * y.size(1)  # Total number of label slots

    accuracy = total_correct / total_samples
    return accuracy, loss_test

best_model=train()