import torch 
import numpy as np 
from matplotlib import pyplot as plt 
import argparse
import os

parser = argparse.ArgumentParser(description='Plotting Parameter Gradients')
parser.add_argument('--dataset-name', type=str, required=True, help='Dataset string for plot')
parser.add_argument('--result-folder', type=str, required=True, help='Filepath for saving image')
parser.add_argument('--checking-hops', action='store_true', help='Are you comparing hops for subgraphs?')
parser.add_argument('--plot-downstream', action='store_true', help='Do you want the downstream task grads plotted?')

args = parser.parse_args()
dataset_name = args.dataset_name
save_path = args.result_folder+'/'
checking_hops = args.checking_hops
plot_downstream = args.plot_downstream 
dataset = args.result_folder.split('/')[-1]

epochs=100 
plotting_dict = {'local_model':'MPNN Backbone', 'self_attn': 'Mamba Block', 'ff_linear': 'Feed Forward Block', 'downstream_layers':'Downstream MLP'}
def diff_dict_location(param_name):
    for key1 in difference_dict.keys():
        if key1 in param_name: 
            for key2 in difference_dict[key1].keys():
                if key2 in param_name:
                    return key1, key2
    return None, None 
def compute_diff(model_i, model_j, param_name):
    param_j = model_j[param_name].float()
    param_i = model_i[param_name].float()
    l2_norm = torch.norm(param_j - param_i, p='fro') # / torch.norm(param_i, p='fro')
    return l2_norm.item()
def plot_dynamics(key1, filename,title,show_downstream=False):
    plt.figure()
    for key2 in difference_dict[key1].keys():
        plt.plot(difference_dict[key1][key2], label=plotting_dict[key2])
    if plot_downstream and show_downstream: 
        plt.plot(difference_dict['downstream_layers']['downstream_layers'], label=plotting_dict['downstream_layers'])
    plt.legend()
    plt.suptitle('Parameter Differences over Epochs')
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm of Difference')
    plt.show()
    plt.ylim((0,y_max_value+5))
    plt.savefig(f'{save_path}{filename}.png')

layers_or_hops=[1,2,4]
if checking_hops: 
    layers_or_hops=[1,2,3]

for k in layers_or_hops: 
    folder_path = f'{save_path}{k}layer_{dataset}/'
    if checking_hops: 
        folder_path = f'{save_path}{k}hop_{dataset}/'
    
    if not os.path.exists(folder_path):
        print(folder_path, 'does not exist.')
        continue

    difference_dict = {
        'attn_layers.0':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
        'attn_layers.1':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
        'attn_layers.2':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
        'attn_layers.3':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
        'downstream_layers':{'downstream_layers':np.zeros(epochs)}
        }

    model_i = torch.load(f'{folder_path}0.pth')
    for j in range(1,epochs):
        try: 
            model_j = torch.load(f'{folder_path}{j}.pth')
            for param_name in model_j:
                key1, key2 = diff_dict_location(param_name)
                if key1!=None and key2!=None: 
                    difference_dict[key1][key2][j]+=compute_diff(model_i, model_j, param_name)
        except: 
            break
    y_max_value = max(max(np.max(difference_dict[layer][block]) for block in difference_dict[layer]) for layer in difference_dict)

    if checking_hops: 
        plot_dynamics(f'attn_layers.0',f'{k}hop_param_diffs',f'{dataset_name}, {k}-Hop model, GMB block 0',show_downstream=True)
    else: 
        for layer in range(k): 
            plot_dynamics(f'attn_layers.{layer}',f'{k}layer_param_diffs_{layer}',f'{dataset_name}, {k}-Layer model, GMB block {layer}',show_downstream=(layer==k-1))


# np.savez('difference_dict.npz', **difference_dict)
# data = np.load('difference_dict.npz', allow_pickle=True)
# difference_dict_loaded = {key: data[key] for key in data.files}

'''
dict_keys(['attn_layers.0.local_model.A.weight', 'attn_layers.0.local_model.A.bias', 'attn_layers.0.local_model.B.weight', 'attn_layers.0.local_model.B.bias', 'attn_layers.0.local_model.C.weight', 'attn_layers.0.local_model.C.bias', 'attn_layers.0.local_model.D.weight', 'attn_layers.0.local_model.D.bias', 'attn_layers.0.local_model.E.weight', 'attn_layers.0.local_model.E.bias', 'attn_layers.0.local_model.bn_node_x.weight', 'attn_layers.0.local_model.bn_node_x.bias', 'attn_layers.0.local_model.bn_edge_e.weight', 'attn_layers.0.local_model.bn_edge_e.bias', 'attn_layers.0.norm1_local.weight', 'attn_layers.0.norm1_local.bias', 'attn_layers.0.norm1_local.mean_scale', 'attn_layers.0.norm1_attn.weight', 'attn_layers.0.norm1_attn.bias', 'attn_layers.0.norm1_attn.mean_scale', 'attn_layers.0.self_attn.A_log', 'attn_layers.0.self_attn.D', 'attn_layers.0.self_attn.in_proj.weight', 'attn_layers.0.self_attn.conv1d.weight', 'attn_layers.0.self_attn.conv1d.bias', 'attn_layers.0.self_attn.x_proj.weight', 'attn_layers.0.self_attn.dt_proj.weight', 'attn_layers.0.self_attn.dt_proj.bias', 'attn_layers.0.self_attn.out_proj.weight', 'attn_layers.0.ff_linear1.weight', 'attn_layers.0.ff_linear1.bias', 'attn_layers.0.ff_linear2.weight', 'attn_layers.0.ff_linear2.bias', 'attn_layers.0.norm2.weight', 'attn_layers.0.norm2.bias', 'attn_layers.0.norm2.mean_scale', 'downstream_layers.0.weight', 'downstream_layers.0.bias', 'downstream_layers.1.weight', 'downstream_layers.1.bias', 'downstream_layers.2.weight', 'downstream_layers.2.bias'])
'''