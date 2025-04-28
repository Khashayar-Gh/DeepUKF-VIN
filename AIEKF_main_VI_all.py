import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  DataLoader
import pickle
torch.set_default_dtype(torch.float64)
from AI_model_utils import vision_noise_model, imu_noise_model, estimation_loss , count_parameters
from AI_model_utils import TimeSeriesGRU2 , CNNGRUModel, CNNModel2, CNNModel3 , TimeSeriesGRU3
from AI_data_utils import CustomDataset , collate_fn_ , CustomDataset_EKF 
import time
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import os
# from torchviz import make_dot , make_dot_from_trace
from EKF_utils import ErrorStateEKF
from torch.utils.checkpoint import checkpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

def print_grad_fn_hook(grad):
    print(f"Gradient from hook: {grad}")

# with open('log_obj_EKF_20240924233405.pkl', 'rb') as file: #9 a_m
# with open('log_obj_EKF_20241001195918.pkl', 'rb') as file: 
#     loaded_data = pickle.load(file)
with open('log_obj_EKF_20241015143753.pkl', 'rb') as file: 
    loaded_data = pickle.load(file)

xhat_k_1_log ,x_k_log ,P_k_1_log , f_B_log ,f_W_log , imu_log ,imu_noise_log , f_n_log = loaded_data.get()
imu_noise_log = [imu_noise_log_.cuda() for imu_noise_log_ in imu_noise_log]
EuRoCReader_obj = loaded_data.EuRoCReader
log_stereo_idx = loaded_data.stereo_idx
f_n_log = [torch.tensor(f_n_) if f_n_ is not None else torch.tensor(0.0) for f_n_ in f_n_log]
imu_s = loaded_data.imu_s
time_s = loaded_data.time
time_ind = loaded_data.time_index

# Create the custom dataset and DataLoader
# seq_length = 9
# seq_length = 2 #for debug
dataset = CustomDataset_EKF(xhat_k_1_log ,P_k_1_log , f_B_log ,f_W_log , imu_s, x_k_log ,
                        EuRoCReader_obj , log_stereo_idx , f_n_log  , include_imgs=True)

dataset_size = len(dataset)
indices = list(range(dataset_size))
# split = int(0.99 * dataset_size) # 10% train data

# train_indices, val_indices = indices[:split], indices[split:]
train_dataset = dataset
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)

# batch_size = len(dataset)
batch_size = 32
dataset_size = len(train_dataset)
# batch_size = 2 #for debug
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False , collate_fn=collate_fn_)
# val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_)

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
current_time = time.strftime("%m %d %H:%M:%S", time.localtime())
logfile_name = f"logfile_AIEKF_Both_{current_time}_all_cri_0_m3.txt"
# Example usage
imu_input_dim = 6  # Number of input features per time step
imu_hidden_dim = 32  # Number of hidden units in the GRU
# imu_hidden_dim = 2  # #for debug
# imu_num_layers = 5  # Number of GRU layers
imu_num_layers = 2  # #for debug
imu_output_dim = 12  # output dim
imu_dropout = 0.5  # Dropout rate

vis_input_dim = 6  # Number of input features per time step
vis_hidden_dim = 32  # Number of hidden units in the GRU
# vis_hidden_dim = 2  # #for debug
# vis_num_layers = 5  # Number of GRU layers
vis_num_layers = 2  # #for debug
vis_out_dim = 1  # output dim
vis_dropout_prob = 0.5  # Dropout rate

# Instantiate models and loss function
# vision_model = TimeSeriesGRU(vis_input_dim , vis_hidden_dim, vis_num_layers, vis_out_dim , vis_dropout_prob).cuda()
# vision_model = CNNGRUModel(vis_hidden_dim, vis_num_layers, vis_out_dim , vis_dropout_prob).cuda()
# vision_model = CNNModel2(vis_out_dim, vis_dropout_prob , vis_hidden_dim).cuda()
vision_model = CNNModel3(vis_out_dim, vis_dropout_prob , vis_hidden_dim).cuda()
# imu_model = TimeSeriesGRU2(imu_input_dim, imu_hidden_dim, imu_num_layers, imu_output_dim , imu_dropout).cuda()
imu_model = TimeSeriesGRU3(imu_input_dim, imu_hidden_dim, imu_num_layers, imu_output_dim , imu_dropout).cuda()
criterion = estimation_loss().cuda()
# mse_loss = nn.MSELoss().cuda()
with open(logfile_name, "a") as logfile:
    logfile.write(f'imu_input_dim = {imu_input_dim} \n')
    logfile.write(f'imu_hidden_dim = {imu_hidden_dim} \n')
    logfile.write(f'imu_num_layers = {imu_num_layers} \n')
    logfile.write(f'imu_output_dim = {imu_output_dim} \n')
    logfile.write(f'imu_dropout = {imu_dropout} \n')
    logfile.write(f'vis_input_dim = {vis_input_dim} \n')
    logfile.write(f'vis_hidden_dim = {vis_hidden_dim} \n')
    logfile.write(f'vis_num_layers = {vis_num_layers} \n')
    logfile.write(f'vis_out_dim = {vis_out_dim} \n')
    logfile.write(f'vis_dropout_prob = {vis_dropout_prob} \n')
    logfile.write(f'vision_model TimeSeriesGRU \n')
    logfile.write(f'imu_model TimeSeriesGRU \n')

# Count the parameters for each model
total_params_cnn, trainable_params_cnn = count_parameters(vision_model)
total_params_imu, trainable_params_deep = count_parameters(imu_model)
print(f"Total parameters for CNN model: {total_params_cnn}, Trainable parameters: {trainable_params_cnn}")
print(f"Total parameters for deep model: {total_params_imu}, Trainable parameters: {trainable_params_deep}")
print(f"current_time = {current_time}")
with open(logfile_name, "a") as logfile:
    logfile.write(f"Total parameters for CNN model: {total_params_cnn}, Trainable parameters: {trainable_params_cnn}\n")
    logfile.write(f"Total parameters for deep model: {total_params_imu}, Trainable parameters: {trainable_params_deep}\n")
# optimizer = optim.Adam(list(vision_model.parameters()) + list(imu_model.parameters()),
#                         lr=0.001 , weight_decay=0.01) #defult lr = 0.001
optimizer = optim.Adam(list(vision_model.parameters()) + list(imu_model.parameters()),
                        lr=0.001 , weight_decay=0.0001) #defult lr = 0.001
with open(logfile_name, "a") as logfile:
    logfile.write(f"weight_decay=: {0.0001}\n")
from QNUKF import QNUKF
# Training loop example
beta = 1.0
tanh = nn.Tanh()
total_epochs = 30
total_batches = len(train_loader)
losses = []
torch.cuda.empty_cache()
optimizer.zero_grad()
start_time = time.time()
xhat_k_1_log_0 = xhat_k_1_log[0].cuda()
P_k_1_log_0 = P_k_1_log[0].cuda()
for epoch in range(total_epochs):  # Number of epochs
    loss_epoch = 0
    # print("epoch = ", epoch)
    # logfile.write(f"Epoch {epoch+1}/{total_epochs}\n")
    filter = ErrorStateEKF(xhat_k_1_log_0 , P_k_1_log_0)
    for j , batch in enumerate(train_loader):
        
        # print("batch number = ", j)

        # batch includes 1 data point (batch_size=1)
        img0 , img1 , xhat_k_1 , P_k_1 , f_B, f_W, imu, x_k , f_n = batch
        # print("loaded the data")
        # img0 = img0[: , -1 ,...]
        # img1 = img1[: , -1 ,...]

        img0 = img0[: , 0 ,...]
        img1 = img1[: , 0 ,...]
        imu_noise_std = imu_model(imu)
        # imu_noise_std = checkpoint(imu_model, imu)

        # print("imu_noise_std = ", imu_noise_std)
        
        batch_size = imu.size(0)
        x = torch.zeros(batch_size , 16 , device='cuda')
        vision_noise_std = vision_model(img0 , img1)
        # vision_noise_std = checkpoint(vision_model, img0, img1)

        
        
        # imu_noise_log[j] = imu_noise_log[j].cuda()
        imu_beta_tanh_outputs =10**(beta*tanh(imu_noise_std))
        vision_beta_tanh_outputs =10**(beta*tanh(vision_noise_std))
        f_n[:][-1] = f_n[:][-1].cuda()
        for i in range(batch_size):
            # print("i = ", i)
            noise_vec_std = imu_noise_log[j]*imu_beta_tanh_outputs[i]
            filter.batch_predict(imu[i][: , :3] , imu[i][: , 3:6] ,  noise_vec_std[0:3] ,
                            noise_vec_std[3:6] , noise_vec_std[6:9] ,
                            noise_vec_std[9:12])
            
            # f_B_ = f_B[i][-1].cuda()
            # f_W_ = f_W[i][-1].cuda()

            f_B_ = f_B[i][0].cuda()
            f_W_ = f_W[i][0].cuda()

            noise_vision_vec_std = f_n[i][-1]*vision_beta_tanh_outputs[i]
            # noise_vision_vec_std = f_n[i][-1].cuda()*10**(beta*tanh(vision_noise_std))
            # print("f_B shape = ", f_B_.shape)
            filter.update(f_W_ , f_B_ ,  noise_vision_vec_std)
            # filter.x = filter.x.detach()
            x[i] = filter.x#IMU train only

        # Compute loss
        # loss = criterion(x, x_k)
        # loss = mse_loss(x, x_k[: , -1 , :])
        loss = criterion(x, x_k[: , 0 , :])
        # loss = mse_loss(x, x_k[: , 0 , :])

        loss_epoch += loss.item()

        loss.backward(retain_graph=True)
        filter.x = filter.x.detach()
        filter.P = filter.P.detach()

        percent_complete = 100 * (epoch * total_batches + j + 1) / (total_epochs * total_batches)

    torch.nn.utils.clip_grad_norm_(vision_model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(imu_model.parameters(), max_norm=1.0)
    optimizer.step()
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    loss_epoch /= total_batches
    losses.append(loss_epoch)
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / (percent_complete/100)
    eta = estimated_total_time - elapsed_time
    # Convert eta to hours and minutes
    eta_days = int(eta // (3600 * 24))
    eta_hours = int((eta % (3600 * 24)) // 3600)
    eta_minutes = int((eta % 3600) // 60)
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Current time: {current_time}")
    elapsed_days = int(elapsed_time // (3600 * 24))
    elapsed_hours = int((elapsed_time % (3600 * 24)) // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {elapsed_days} days, {elapsed_hours} hours, {elapsed_minutes} minutes, {elapsed_seconds} seconds")
    print(f"Completion: {percent_complete:.2f}%")
    print(f"ETA: {eta_days} days, {eta_hours} hours and {eta_minutes} minutes")

    print(f'Epoch [{epoch+1}/10], Loss: {loss_epoch}')
    with open(logfile_name, "a") as logfile:
        logfile.write(f'Epoch [{epoch+1}/10], Loss: {loss_epoch}\n')
    # Plot the loss curve
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')  # Save the plot

current_time = time.strftime("%m %d %H:%M:%S", time.localtime())
torch.save(vision_model.state_dict(), f'vision_model_{current_time}.pth')
torch.save(imu_model.state_dict(),    f'imu_model_{current_time}.pth')
with open(logfile_name, "a") as logfile:
    logfile.write(f'vision_model_{current_time}.pth\n')
    logfile.write(f'imu_model_{current_time}.pth\n')
print("log file name = ", logfile_name)