import numpy as np
import re
from matplotlib import pyplot as plt 

# Define the log file path
folder_path = 'what_did_we_learn/apr21-b/hpo_metab/'
checking_hops=False
if checking_hops: 
    hops=1
    filename=f'{hops}hop_hpo_metab'
    description = f'{hops}-hop neighborhood for HPO Metab'
else:
    layers=4
    filename=f'{layers}layer_hpo_metab'
    description = f'{layers}-layer GMB model for HPO Metab'
file_path = f'{folder_path}{filename}.log'

# Initialize lists to store the parsed data
epochs = []
loss_train = []
acc_train = []
acc_val = []

# Compile the regular expression for matching lines with the required data
pattern = re.compile(r'Epoch: (\d{4}) loss_train: ([\d.]+) acc_train: ([\d.]+) acc_val: ([\d.]+)')

# Read the log file and parse the data
with open(file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            # Append the data to the respective lists after converting to the correct type
            epochs.append(int(match.group(1)))
            loss_train.append(float(match.group(2)))
            acc_train.append(float(match.group(3)))
            acc_val.append(float(match.group(4)))

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

