# import packages
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 최신 CPU라 더 빨라질수 있다는데.. 어차피 CPU연산 안함
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import torch
from basicsr.models.archs.restormer_arch import Restormer
from pdb import set_trace as stx
import gc
import tensorflow as tf
from util import *

# GPU 할당량 최대치의 60%로 제한 => 발열 제어
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)

##########################################
# 01. LOAD YAML
##########################################
yaml_file = 'Options/RealDenoising_Restormer_small.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r', encoding='UTF8'), Loader=Loader)

s = x['network_g'].pop('type')


##########################################
# 02. LOAD MODEL AND WEIGHT
##########################################
model_restoration = Restormer(**x['network_g'])
model_restoration.train()

# first step
# checkpoint = torch.load('./pretrained_models/real_denoising.pth')
# model_restoration.load_state_dict(checkpoint['params'], strict=False)

# second step
model_restoration.load_state_dict(torch.load('./pts/2_9fine_tuned_model12.pt'))

model_restoration.cuda()

# 모델 전체 프리즈 시키고
for param in model_restoration.parameters():
    param.requires_grad = False

# model.named_parameters() 로 name을 확인하고 아래와 같이 required_grad = True 로 변경
for name, param in model_restoration.named_parameters():
    if 'attn' in name :
        param.requires_grad = True
        print(name)


##########################################
# 03. PREPARE DATA
##########################################
step = 3

def make_DataPair(step, patch_size, image_nums, batch_size, val=False):
    # 1) folder to patches
    with tf.device('/device:GPU:0'):
        #  8163 / 100 = 80
        #  2040 / 25 = 80
        step = step #[0, 40]
        if val:
            X_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/val/origin/', patch_size, step, size=image_nums)
            y_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/val/gaussian/',patch_size, step, size=image_nums)
        else:
            X_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/train/origin/', patch_size, step, size=image_nums)
            y_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/train/gaussian/', patch_size, step, size=image_nums)

        # 2) Make Sequence
        X_Patches = to_sequence(X_Patches)
        y_Patches = to_sequence(y_Patches)
        print("Patches.shape : ", X_Patches.shape) # (n, 256, 256, 3)

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
train_dataloader.append(make_DataPair(step=step, patch_size=32, image_nums=8, batch_size=8))
train_dataloader.append(make_DataPair(step=step, patch_size=48, image_nums=8, batch_size=4))
train_dataloader.append(make_DataPair(step=step, patch_size=64, image_nums=8, batch_size=2))
train_dataloader.append(make_DataPair(step=step, patch_size=96, image_nums=8, batch_size=1))

val_dataloader = make_DataPair(step=step, patch_size=96, image_nums=2, batch_size=1, val=True)


##########################################
# 04. LEARNING
##########################################
checkpoint_path = "restormer/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Define the loss function
criterion = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model_restoration.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-4) # lr = 3e-4, betas=(0.9, 0.999), weight_decay=1e-4                      
num_epochs = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8) # eta_min = 1e-6

# Fine-tune the model on your data
for epoch in range(num_epochs):
    
    # 이어서
    # if epoch < 30:
    #    continue
    
    train_loss = 0.0
    model_restoration.train()  # Set the model to training mode
    
    for progressive_learning in range(4):
        for batch_idx, (target, data) in enumerate(train_dataloader[progressive_learning]):
            data = torch.tensor(data, requires_grad=True)  # Convert data to PyTorch tensor
            target = torch.tensor(target, requires_grad=True)  # Convert target to PyTorch tensor
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            
            data = data.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)
            
            output = model_restoration(data.float())
            copied_output = output
            loss = criterion(copied_output, target.float())
            
            loss.backward()
            optimizer.step()
            
            # Scheduler update (for learning rate adjustment)
            scheduler.step()
            
            train_loss += loss.item()
            del data, target, output, copied_output
            
            print("Training ..[", progressive_learning, "]...", batch_idx, " / " , len(train_dataloader[progressive_learning]), end='                                    \r')
            

    # Validation loop (similar modifications as in the training loop)
    model_restoration.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (target, data) in enumerate(val_dataloader):
            data = torch.tensor(data, requires_grad=False)
            target = torch.tensor(target, requires_grad=False)

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            data = data.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

            output = model_restoration(data.float())
            copied_output = output
            val_loss += criterion(copied_output, target.float())
            del data, target, output, copied_output
            

    val_loss /= len(val_dataloader)
    train_loss /= (len(train_dataloader[0]) + len(train_dataloader[1]) + len(train_dataloader[2]) + len(train_dataloader[3]))

    gc.collect()
    torch.cuda.empty_cache()
    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.60f}, Val Loss: {val_loss:.60f}')

    # Save the fine-tuned model
    torch.save(model_restoration.state_dict(), './pts/'+str(step)+'_'+str(epoch)+'fine_tuned_model12.pt')
