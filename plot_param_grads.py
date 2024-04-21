import torch 
import numpy as np 
from matplotlib import pyplot as plt 

epochs=100 

n_layers=1
n_hops = 1
checking_hops = True 
save_path = 'what_did_we_learn/apr21-c/hpo_metab/'
dataset_name='HPO Metab'

folder_path = f'{save_path}{n_layers}layer_hpo_metab/'
if checking_hops: 
    folder_path = f'{save_path}{n_hops}hop_hpo_metab/'

gradient_dict = {
    'attn_layers.0':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
    'attn_layers.1':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
    'attn_layers.2':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)},
    'attn_layers.3':{'local_model':np.zeros(epochs), 'self_attn':np.zeros(epochs), 'ff_linear':np.zeros(epochs)}}

plotting_dict = {'local_model':'MPNN Backbone', 'self_attn': 'Mamba Block', 'ff_linear': 'Feed Forward Block'}

def dict_location(param_name):
    if 'local_model.bn_edge_e.' in param_name: # because edge attributes are all 0
        return None, None 
    for key1 in gradient_dict.keys():
        if key1 in param_name: 
            for key2 in gradient_dict[key1].keys():
                if key2 in param_name:
                    return key1, key2
    return None, None 

for epoch in range(epochs): 
    batch = 0
    get_file=True
    while get_file: 
        try: 
            filename = f'{folder_path}gradients_{epoch}_{batch}.pth'
            gradients = torch.load(filename)
            for param_name in gradients.keys(): 
                key1, key2 = dict_location(param_name)
                if key1 is not None and key2 is not None: 
                    grads = gradients[param_name].float() 
                    gradient_dict[key1][key2][epoch]+=torch.norm(grads, p='fro').item() 
            batch+=1
        except FileNotFoundError:
            get_file=False

# just creating cutoff value for plotting 
for key1 in gradient_dict.keys():
    for key2 in gradient_dict[key1].keys(): 
        gradient_dict[key1][key2] = np.clip(gradient_dict[key1][key2], None, 1e7)

def plot_dynamics(key1, filename,title):
    plt.figure()
    for key2 in gradient_dict[key1].keys():
        plt.plot(gradient_dict[key1][key2], label=plotting_dict[key2])
    plt.legend()
    plt.suptitle('Gradient Norms over Epochs')
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Frobenius Norm')
    plt.show()
    # plt.ylim((0,y_max_value+5))
    plt.savefig(f'{save_path}{filename}.png')

if checking_hops: 
    plot_dynamics(f'attn_layers.0',f'{n_hops}hop_gradients',f'{dataset_name}, {n_hops}-Hop model, GMB block 0')
else:
    for layer in range(n_layers): 
        plot_dynamics(f'attn_layers.{layer}',f'{n_layers}layer_gradients_{layer}',f'{dataset_name}, {n_layers}-Layer model, GMB block {layer}')

'''
# graph convolutions 
attn_layers.0.local_model.A.weight       torch.Size([64, 64])
attn_layers.0.local_model.A.bias         torch.Size([64])
attn_layers.0.local_model.B.weight       torch.Size([64, 64])
attn_layers.0.local_model.B.bias         torch.Size([64])
attn_layers.0.local_model.C.weight       torch.Size([64, 64])
attn_layers.0.local_model.C.bias         torch.Size([64])
attn_layers.0.local_model.D.weight       torch.Size([64, 64])
attn_layers.0.local_model.D.bias         torch.Size([64])
attn_layers.0.local_model.E.weight       torch.Size([64, 64])
attn_layers.0.local_model.E.bias         torch.Size([64])
# normalization 
attn_layers.0.local_model.bn_node_x.weight       torch.Size([64])
attn_layers.0.local_model.bn_node_x.bias         torch.Size([64])
attn_layers.0.local_model.bn_node_x.running_mean         torch.Size([64])
attn_layers.0.local_model.bn_node_x.running_var          torch.Size([64])
attn_layers.0.local_model.bn_node_x.num_batches_tracked          torch.Size([])
attn_layers.0.local_model.bn_edge_e.weight       torch.Size([64])
attn_layers.0.local_model.bn_edge_e.bias         torch.Size([64])
attn_layers.0.local_model.bn_edge_e.running_mean         torch.Size([64])
attn_layers.0.local_model.bn_edge_e.running_var          torch.Size([64])
attn_layers.0.local_model.bn_edge_e.num_batches_tracked          torch.Size([])
# normalization 
attn_layers.0.norm1_local.weight         torch.Size([64])
attn_layers.0.norm1_local.bias   torch.Size([64])
attn_layers.0.norm1_local.mean_scale     torch.Size([64])
attn_layers.0.norm1_attn.weight          torch.Size([64])
attn_layers.0.norm1_attn.bias    torch.Size([64])
attn_layers.0.norm1_attn.mean_scale      torch.Size([64])
# mamba 
attn_layers.0.self_attn.A_log    torch.Size([64, 16])
attn_layers.0.self_attn.D        torch.Size([64])
attn_layers.0.self_attn.in_proj.weight   torch.Size([128, 64])
attn_layers.0.self_attn.conv1d.weight    torch.Size([64, 1, 4])
attn_layers.0.self_attn.conv1d.bias      torch.Size([64])
attn_layers.0.self_attn.x_proj.weight    torch.Size([36, 64])
attn_layers.0.self_attn.dt_proj.weight   torch.Size([64, 4])
attn_layers.0.self_attn.dt_proj.bias     torch.Size([64])
attn_layers.0.self_attn.out_proj.weight          torch.Size([64, 64])
# feed forward 
attn_layers.0.ff_linear1.weight          torch.Size([128, 64])
attn_layers.0.ff_linear1.bias    torch.Size([128])
attn_layers.0.ff_linear2.weight          torch.Size([64, 128])
attn_layers.0.ff_linear2.bias    torch.Size([64])
attn_layers.0.norm2.weight       torch.Size([64])
attn_layers.0.norm2.bias         torch.Size([64])
attn_layers.0.norm2.mean_scale   torch.Size([64])

attn_layers.1.local_model.A.weight       torch.Size([64, 64])
attn_layers.1.local_model.A.bias         torch.Size([64])
attn_layers.1.local_model.B.weight       torch.Size([64, 64])
attn_layers.1.local_model.B.bias         torch.Size([64])
attn_layers.1.local_model.C.weight       torch.Size([64, 64])
attn_layers.1.local_model.C.bias         torch.Size([64])
attn_layers.1.local_model.D.weight       torch.Size([64, 64])
attn_layers.1.local_model.D.bias         torch.Size([64])
attn_layers.1.local_model.E.weight       torch.Size([64, 64])
attn_layers.1.local_model.E.bias         torch.Size([64])
attn_layers.1.local_model.bn_node_x.weight       torch.Size([64])
attn_layers.1.local_model.bn_node_x.bias         torch.Size([64])
attn_layers.1.local_model.bn_node_x.running_mean         torch.Size([64])
attn_layers.1.local_model.bn_node_x.running_var          torch.Size([64])
attn_layers.1.local_model.bn_node_x.num_batches_tracked          torch.Size([])
attn_layers.1.local_model.bn_edge_e.weight       torch.Size([64])
attn_layers.1.local_model.bn_edge_e.bias         torch.Size([64])
attn_layers.1.local_model.bn_edge_e.running_mean         torch.Size([64])
attn_layers.1.local_model.bn_edge_e.running_var          torch.Size([64])
attn_layers.1.local_model.bn_edge_e.num_batches_tracked          torch.Size([])
attn_layers.1.norm1_local.weight         torch.Size([64])
attn_layers.1.norm1_local.bias   torch.Size([64])
attn_layers.1.norm1_local.mean_scale     torch.Size([64])
attn_layers.1.norm1_attn.weight          torch.Size([64])
attn_layers.1.norm1_attn.bias    torch.Size([64])
attn_layers.1.norm1_attn.mean_scale      torch.Size([64])
attn_layers.1.self_attn.A_log    torch.Size([64, 16])
attn_layers.1.self_attn.D        torch.Size([64])
attn_layers.1.self_attn.in_proj.weight   torch.Size([128, 64])
attn_layers.1.self_attn.conv1d.weight    torch.Size([64, 1, 4])
attn_layers.1.self_attn.conv1d.bias      torch.Size([64])
attn_layers.1.self_attn.x_proj.weight    torch.Size([36, 64])
attn_layers.1.self_attn.dt_proj.weight   torch.Size([64, 4])
attn_layers.1.self_attn.dt_proj.bias     torch.Size([64])
attn_layers.1.self_attn.out_proj.weight          torch.Size([64, 64])
attn_layers.1.ff_linear1.weight          torch.Size([128, 64])
attn_layers.1.ff_linear1.bias    torch.Size([128])
attn_layers.1.ff_linear2.weight          torch.Size([64, 128])
attn_layers.1.ff_linear2.bias    torch.Size([64])
attn_layers.1.norm2.weight       torch.Size([64])
attn_layers.1.norm2.bias         torch.Size([64])
attn_layers.1.norm2.mean_scale   torch.Size([64])
fc2.weight       torch.Size([6, 64])
fc2.bias         torch.Size([6])
'''