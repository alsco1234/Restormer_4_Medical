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
import time

# GPU 할당량 최대치의 60%로 제한 => 발열 제어
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)

##########################################
# 01. LOAD YAML
##########################################
yaml_file = 'Options/RealDenoising_Restormer.yml'
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
checkpoint = torch.load('./pretrained_models/real_denoising.pth')
model_restoration.load_state_dict(checkpoint['params'])

# second step
# model_restoration.load_state_dict(torch.load('./pts/0_29fine_tuned_model7.pt'))

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
# 1) folder to patches
with tf.device('/device:GPU:0'):
    #  8163 / 100 = 80
    #  2040 / 25 = 80
    step = 0 #[0, 40]
    X_Train_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/train/origin/', 48, step, size=8)
    y_Train_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/train/gaussian/', 48, step, size=8)
    X_Val_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/val/origin/', 48, step, size=2)
    y_Val_Patches = folder_to_patches('C:/Users/NDT/Desktop/Image_denoising/Data/val/gaussian/',48, step, size=2)

    # 2) Make Sequence
    X_Train_Patches = to_sequence(X_Train_Patches)
    print("X_Train_Patches.shape : ", X_Train_Patches.shape) # (n, 256, 256, 3)
    y_Train_Patches = to_sequence(y_Train_Patches)
    print("y_Train_Patches.shape : ", y_Train_Patches.shape) # (n, 256, 256, 3)
    X_Val_Patches = to_sequence(X_Val_Patches)
    print("X_Val_Patches.shape : ", X_Val_Patches.shape) # (n, 256, 256, 3)
    y_Val_Patches = to_sequence(y_Val_Patches)
    print("y_Val_Patches.shape : ", y_Val_Patches.shape) # (n, 256, 256, 3)

    # 3) Normalization
    X_Train_Data = np.array(X_Train_Patches, dtype = object).astype(float) / 255.0
    print("finising normalizate X_Train_Data")
    y_Train_Data = np.array(y_Train_Patches, dtype = object).astype(float) / 255.0
    print("finising normalizate y_Train_Data")
    X_Val_Data = np.array(X_Val_Patches, dtype = object).astype(float) / 255.0
    print("finising normalizate X_Val_Data")
    y_Val_Data = np.array(y_Val_Patches, dtype = object).astype(float) / 255.0
    print("finising normalizate y_Val_Data")

del X_Train_Patches, y_Train_Patches, X_Val_Patches, y_Val_Patches
gc.collect()
torch.cuda.empty_cache()

# 4) Tie Ground_Truth and Noisy
batch_size = 4
train_dataloader = Dataloder(X_Train_Data, y_Train_Data, batch_size, shuffle=True)
val_dataloader = Dataloder(X_Val_Data, y_Val_Data, batch_size, shuffle=True)

del X_Train_Data, y_Train_Data, X_Val_Data, y_Val_Data
gc.collect()
torch.cuda.empty_cache()


##########################################
# 04. LEARNING
##########################################
checkpoint_path = "restormer/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Define the loss function
criterion = torch.nn.L1Loss()

# Define the optimizer (weight_decaay = L2 regularization parameter)
optimizer = torch.optim.AdamW(model_restoration.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-4) # lr = 3e-4, betas=(0.9, 0.999), weight_decay=1e-4                      

num_epochs = 50
# Define the cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8) # eta_min = 1e-6

# Fine-tune the model on your data
for epoch in range(num_epochs):
    
    # 이어서
    # if epoch < 30:
    #    continue
    
    train_loss = 0.0
    model_restoration.train()  # Set the model to training mode
    
    for batch_idx, (target, data) in enumerate(train_dataloader):
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
        
        print("Training ..", batch_idx, " / " , len(train_dataloader), end='                                    \r')
        
        # GPU temperature
        time.sleep(0.5)

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
            
            # GPU temperature
            time.sleep(1.0)

    val_loss /= len(val_dataloader)
    train_loss /= len(train_dataloader)

    gc.collect()
    torch.cuda.empty_cache()
    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.60f}, Val Loss: {val_loss:.60f}')

    # Save the fine-tuned model
    torch.save(model_restoration.state_dict(), './pts/'+str(step)+'_'+str(epoch)+'fine_tuned_model10.pt')
