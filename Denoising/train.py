import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from basicsr.models.archs.restormer_arch import Restormer
import gc
import tensorflow as tf
import argparse
from utils import folder_to_patches, to_sequence, Dataloder
import time
import yaml
import datetime

str_time = time.time()

parser = argparse.ArgumentParser(description='Grayscale Denoising using fine-tuned Restormer')

parser.add_argument('--step', default=0, type=int, help='Path of noisy images')
parser.add_argument('--test_origin_folder', default='./data/test/origin/', type=str, help='Path of test origin folder')
parser.add_argument('--test_noisy_folder', default='./data/test/noisy/', type=str, help='Path of test noisy folder')
parser.add_argument('--val_origin_folder', default='./data/val/origin/', type=str, help='Path of validation origin folder')
parser.add_argument('--val_noisy_folder', default='./data/val/noisy/', type=str, help='Path of  validation noisy folder')
parser.add_argument('--pretrained', default=False, type=bool, help='If you want to test pretrained, not fine tuned weight')
args = parser.parse_args()


##########################################
# 01. PREPARE DATA
##########################################
step = args.step

def make_DataPair(step, patch_size, image_nums, batch_size, val=False):
    # 1) folder to patches
    with tf.device('/device:GPU:0'):
        if val:
            X_Patches = folder_to_patches(args.test_origin_folder, patch_size, step, size=image_nums)
            y_Patches = folder_to_patches(args.test_noisy_folder, patch_size, step, size=image_nums)
        else:
            X_Patches = folder_to_patches(args.val_origin_folder, patch_size, step, size=image_nums)
            y_Patches = folder_to_patches(args.val_origin_folder, patch_size, step, size=image_nums)

        # 2) Make Sequence
        X_Patches = to_sequence(X_Patches)
        y_Patches = to_sequence(y_Patches)

        # 3) Normalization
        X_Patches = np.array(X_Patches, dtype = object).astype(float) / 255.0
        y_Patches = np.array(y_Patches, dtype = object).astype(float) / 255.0

    # 4) Tie Ground_Truth and Noisy
    batch_size = batch_size
    _dataloader = Dataloder(X_Patches, y_Patches, batch_size, shuffle=True)

    del X_Patches, y_Patches
    gc.collect()
    torch.cuda.empty_cache()
    
    return _dataloader

# make dataloader for 4 training and 1 validation
train_dataloader = []
train_dataloader.append(make_DataPair(step=step, patch_size=32, image_nums=4, batch_size=8))
train_dataloader.append(make_DataPair(step=step, patch_size=48, image_nums=4, batch_size=4))
train_dataloader.append(make_DataPair(step=step, patch_size=64, image_nums=4, batch_size=2))
train_dataloader.append(make_DataPair(step=step, patch_size=96, image_nums=4, batch_size=1))

val_dataloader = make_DataPair(step=step, patch_size=96, image_nums=1, batch_size=1, val=True)


##########################################
# 02. LOAD MODEL
##########################################
# 1) Set Yaml
yaml_file = 'options/RealDenoising_Restormer_small.yml'

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r', encoding='UTF8'), Loader=Loader)
s = x['network_g'].pop('type')


# 2) Load Model and Weight
model_restoration = Restormer(**x['network_g'])
model_restoration.train()

if args.pretrained:
    model_restoration.load_state_dict(torch.load('./pts/Pretrained.pt')['params'], strict=False)
else:
    model_restoration.load_state_dict(torch.load('./pts/Finetuned.pt')) 

model_restoration.cuda()


# 3) Freeze and Unfreeze layers
for param in model_restoration.parameters():
    param.requires_grad = False

for name, param in model_restoration.named_parameters():
    if 'attn' in name :
        param.requires_grad = True
        print(name)


##########################################
# 03. LEARNING
##########################################
# 1) Set hyperparameters
checkpoint_path = "pts"
checkpoint_dir = os.path.dirname(checkpoint_path)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model_restoration.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-4) # lr = 3e-4, betas=(0.9, 0.999), weight_decay=1e-4                      
num_epochs = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8) # eta_min = 1e-6


# 2) Train the model
for epoch in range(num_epochs):
    
    # ===== Train Loop ==== #
    train_loss = 0.0
    model_restoration.train()
    
    for progressive_learning in range(4):
        for batch_idx, (target, data) in enumerate(train_dataloader[progressive_learning]):
            data = torch.tensor(data, requires_grad=True)  # Convert data to PyTorch tensor
            target = torch.tensor(target, requires_grad=True)  # Convert target to PyTorch tensor
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            
            output = model_restoration(data.float())
            copied_output = output
            loss = criterion(copied_output, target.float())
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            print("Training ..[", progressive_learning, "]...", batch_idx, " / " , len(train_dataloader[progressive_learning]), end='                                    \r')

    # ===== Validation Loop ==== #
    val_loss = 0.0
    model_restoration.eval()
    
    with torch.no_grad():
        for batch_idx, (target, data) in enumerate(val_dataloader):
            data = torch.tensor(data, requires_grad=False)
            target = torch.tensor(target, requires_grad=False)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            data = data.unsqueeze(1)
            target = target.unsqueeze(1)

            output = model_restoration(data.float())
            copied_output = output
            val_loss += criterion(copied_output, target.float())
            
    val_loss /= len(val_dataloader)
    train_loss /= (len(train_dataloader[0]) + len(train_dataloader[1]) + len(train_dataloader[2]) + len(train_dataloader[3]))

    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.60f}, Val Loss: {val_loss:.60f}')

# 3) Save Fine-tuned Weight
    torch.save(model_restoration.state_dict(), './pts/'+str(step)+'_'+str(epoch)+'fine_tuned_model.pt')
    

# 4) Print Learning Time  
end_time = time.time
time_difference = end_time - str_time
time_duration = datetime.timedelta(seconds=time_difference)
print("Learning time:", time_duration)