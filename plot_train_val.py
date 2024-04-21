import re
import torch 
import numpy as np 
from matplotlib import pyplot as plt 
import argparse
parser = argparse.ArgumentParser(description='Plotting Parameter Gradients')
parser.add_argument('--dataset-name', type=str, required=True, help='Dataset string for plot')
parser.add_argument('--result-folder', type=str, required=True, help='Filepath for saving image')
parser.add_argument('--checking-hops', action='store_true', help='Are you comparing hops for subgraphs?')

args = parser.parse_args()
dataset_name = args.dataset_name
folder_path = args.result_folder+'/'
checking_hops = args.checking_hops
dataset = args.result_folder.split('/')[-1]

layers_or_hops=[1,2,4]
if checking_hops: 
    layers_or_hops=[1,2,3]

for k in layers_or_hops: 
    if checking_hops:
        filename = f'{k}hop_{dataset}'
        description = f'{k}-hop neighborhood for {dataset_name}'
    else:
        filename = f'{k}layer_{dataset}'
        description = f'{k}-layer GMB model for {dataset_name}'
    file_path = f'{folder_path}{filename}.log'

    # Initialize lists to store the parsed data
    epochs = []
    loss_train = []
    acc_train = []
    acc_val = []

    # Compile the regular expression for matching lines with the required data
    pattern = re.compile(r'Epoch: (\d{4}) loss_train: ([\d.]+) acc_train: ([\d.]+) acc_val: ([\d.]+)')

    # Read the log file and parse the data
    try: 
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    # Append the data to the respective lists after converting to the correct type
                    epochs.append(int(match.group(1)))
                    loss_train.append(float(match.group(2)))
                    acc_train.append(float(match.group(3)))
                    acc_val.append(float(match.group(4)))
    except: 
        continue 
    # Convert lists to NumPy arrays
    epochs_array = np.array(epochs)
    loss_train_array = np.array(loss_train)
    acc_train_array = np.array(acc_train)
    acc_val_array = np.array(acc_val)

    # Plot the training and validation accuracy by epochs
    plt.figure()
    plt.plot(epochs_array, acc_train_array, label='Training Accuracy')
    plt.plot(epochs_array, acc_val_array, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.suptitle(f'Training and Validation Accuracy by Epoch')
    plt.title(f'{description}')
    plt.legend()
    plt.ylim((0,1))
    plt.grid(True)
    plt.savefig(f'{folder_path}{filename}-accuracy.png')

    # Plot the training loss by epochs
    plt.figure()
    plt.plot(epochs_array, loss_train_array, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.suptitle(f'Training Loss by Epoch')
    plt.title(f'{description}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder_path}{filename}-loss.png')

