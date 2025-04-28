import torch
from torch.utils.data import Dataset, DataLoader
# import pickle
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
import numpy as np

torch.set_default_dtype(torch.float64)
# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, xhat_k_1_log ,P_k_1_log , f_B_log ,f_W_log , imu_log, x_k_log ,
                  EuRoCReader_obj , log_stereo_idx , f_n_log , seq_length=1 , device = 'cuda', include_imgs = True):
        self.xhat_k_1_log = xhat_k_1_log
        self.P_k_1_log = P_k_1_log
        self.f_B_log = f_B_log
        self.f_W_log = f_W_log
        self.imu_log = imu_log
        self.x_k_log = x_k_log
        self.image_data_cam0_sync= EuRoCReader_obj.image_data_cam0_sync
        self.image_data_cam1_sync= EuRoCReader_obj.image_data_cam1_sync
        self.log_stereo_idx = log_stereo_idx
        self.f_n_log = f_n_log
        self.seq_length = seq_length
        self.device = device
        self.include_imgs = include_imgs
        

    def __len__(self):
        return len(self.xhat_k_1_log)- self.seq_length + 1

    def __getitem__(self, idx):
        xhat_k_1 = torch.stack(self.xhat_k_1_log[idx:idx + self.seq_length]).to(self.device)
        P_k_1 = torch.stack(self.P_k_1_log[idx:idx + self.seq_length]).to(self.device)
        f_B_s = self.f_B_log[idx:idx + self.seq_length]
        f_W_s = self.f_W_log[idx:idx + self.seq_length]
        imu = torch.tensor(np.array(self.imu_log[idx:idx + self.seq_length])).to(self.device)
        x_k = torch.stack(self.x_k_log[idx:idx + self.seq_length]).to(self.device)
        f_n = torch.stack(self.f_n_log[idx:idx + self.seq_length]).to(self.device)
        img0_list = []
        img1_list = []
        for i, f_B in enumerate(f_B_s):
            if f_B is None:
                if self.include_imgs:
                    if self.log_stereo_idx[idx + i] < self.log_stereo_idx[idx + i + 1]:
                        # img exists but feature points don't exist
                        img_indx = self.log_stereo_idx[idx + i]
                        img0 = read_image(str(self.image_data_cam0_sync[img_indx]),
                                        ImageReadMode.GRAY).type(torch.float64).to(self.device)
                        img1 = read_image(str(self.image_data_cam1_sync[img_indx]),
                                        ImageReadMode.GRAY).type(torch.float64).to(self.device)
                    else:
                        img0 = torch.zeros((1, 480, 752)).to(self.device)
                        img1 = torch.zeros((1, 480, 752)).to(self.device)
                f_B_s[i] = torch.zeros((3, 1)).to(self.device)
                f_W_s[i] = torch.zeros((3, 1)).to(self.device)
            else:
                f_B_s[i] = f_B_s[i].to(self.device)
                f_W_s[i] = f_W_s[i].to(self.device)
                img_indx = self.log_stereo_idx[idx + i]
                if self.include_imgs:
                    img0 = read_image(str(self.image_data_cam0_sync[img_indx]),
                                    ImageReadMode.GRAY).type(torch.float64).to(self.device)
                    img1 = read_image(str(self.image_data_cam1_sync[img_indx]),
                                    ImageReadMode.GRAY).type(torch.float64).to(self.device)
            if self.include_imgs:
                img0_list.append(img0 / 255.00)
                img1_list.append(img1 / 255.00)
        if self.include_imgs:
            img0_list = torch.stack(img0_list)
            img1_list = torch.stack(img1_list)
        else:
            img0_list = xhat_k_1
            img1_list = xhat_k_1
        return img0_list, img1_list, xhat_k_1, P_k_1, f_B_s, f_W_s, imu, x_k, f_n

    # def __getitem__(self, idx):
    #     xhat_k_1 = torch.stack(self.xhat_k_1_log[idx:idx + self.seq_length])
    #     P_k_1 = torch.stack(self.P_k_1_log[idx:idx + self.seq_length])
    #     f_B_s = self.f_B_log[idx:idx + self.seq_length]
    #     f_W_s = self.f_W_log[idx:idx + self.seq_length]
    #     imu = torch.tensor(np.array(self.imu_log[idx:idx + self.seq_length]))
    #     x_k = torch.stack(self.x_k_log[idx:idx + self.seq_length])
    #     f_n = torch.stack(self.f_n_log[idx:idx + self.seq_length])
    #     img0_list = []
    #     img1_list = []
    #     for i , f_B in enumerate(f_B_s):
    #         if f_B is None:
    #             if self.log_stereo_idx[idx + i] < self.log_stereo_idx[idx + i+1]:
    #                 # img exists but feature points don't exist
    #                 img_indx = self.log_stereo_idx[idx + i]
    #                 img0 = read_image(str(self.image_data_cam0_sync[img_indx]),
    #                                    ImageReadMode.GRAY).type(torch.float64).cuda()
    #                 img1 = read_image(str(self.image_data_cam1_sync[img_indx]),
    #                                    ImageReadMode.GRAY).type(torch.float64).cuda()
    #             else:
    #                 img0 = torch.zeros((1, 480, 752)).cuda()
    #                 img1 = torch.zeros((1, 480, 752)).cuda()
    #             f_B_s[i]  = torch.zeros((3 , 1)).cuda()
    #             f_W_s[i]  = torch.zeros((3 , 1)).cuda()
    #             # f_B = torch.tensor([]).cuda()
    #             # f_W = torch.tensor([]).cuda()
    #         else:
    #             # img0 = read_image(str(self.image_data_cam0_sync[self.log_stereo_idx[idx:idx + 
    #             #  self.seq_length]]), ImageReadMode.GRAY).type(torch.float64).cuda()
    #             # img1 = read_image(str(self.image_data_cam1_sync[self.log_stereo_idx[idx:idx + 
    #             # self.seq_length]]), ImageReadMode.GRAY).type(torch.float64).cuda()
    #             # Initialize lists to hold processed images
    #             f_B_s[i] = f_B_s[i].cuda()
    #             f_W_s[i] = f_W_s[i].cuda()
    #             img_indx = self.log_stereo_idx[idx + i]
    #             img0 = read_image(str(self.image_data_cam0_sync[img_indx]),
    #                                ImageReadMode.GRAY).type(torch.float64).cuda()
    #             img1 = read_image(str(self.image_data_cam1_sync[img_indx]),
    #                                ImageReadMode.GRAY).type(torch.float64).cuda()
    #         img0_list.append(img0/255.00)
    #         img1_list.append(img1/255.00)
    #     img0_list = torch.stack(img0_list)
    #     img1_list = torch.stack(img1_list)
    #     # f_B_s = torch.stack(f_B_s)
    #     # f_W_s = torch.stack(f_W_s)
    #     return img0_list , img1_list , xhat_k_1.cuda(), P_k_1.cuda(), f_B_s, f_W_s, \
    #           imu.cuda(), x_k.cuda() , f_n.cuda()

class CustomDataset_EKF(Dataset):
    def __init__(self, xhat_k_1_log ,P_k_1_log , f_B_log ,f_W_log , imu_s, x_k_log ,
                  EuRoCReader_obj , log_stereo_idx , f_n_log , seq_length=10 , device = 'cuda', include_imgs = True):
        self.xhat_k_1_log = xhat_k_1_log
        self.P_k_1_log = P_k_1_log
        self.f_B_log = f_B_log
        self.f_W_log = f_W_log
        self.imu_s = imu_s
        self.x_k_log = x_k_log
        self.image_data_cam0_sync= EuRoCReader_obj.image_data_cam0_sync
        self.image_data_cam1_sync= EuRoCReader_obj.image_data_cam1_sync
        self.log_stereo_idx = log_stereo_idx
        self.f_n_log = f_n_log
        self.seq_length = seq_length
        self.device = device
        self.include_imgs = include_imgs
        

    def __len__(self):
        return len(self.xhat_k_1_log)- self.seq_length + 1

    def __getitem__(self, idx):
        xhat_k_1 = torch.stack(self.xhat_k_1_log[idx:idx + self.seq_length]).to(self.device)
        P_k_1 = torch.stack(self.P_k_1_log[idx:idx + self.seq_length]).to(self.device)
        f_B_s = self.f_B_log[idx:idx + self.seq_length]
        f_W_s = self.f_W_log[idx:idx + self.seq_length]
        imu = torch.tensor(np.array(self.imu_s[idx][-self.seq_length:])).to(self.device)
        x_k = torch.stack(self.x_k_log[idx:idx + self.seq_length]).to(self.device)
        f_n = torch.stack(self.f_n_log[idx:idx + self.seq_length]).to(self.device)
        img0_list = []
        img1_list = []
        for i, f_B in enumerate(f_B_s):
            if f_B is None:
                if self.include_imgs:
                    if self.log_stereo_idx[idx + i] < self.log_stereo_idx[idx + i + 1]:
                        # img exists but feature points don't exist
                        img_indx = self.log_stereo_idx[idx + i]
                        img0 = read_image(str(self.image_data_cam0_sync[img_indx]),
                                        ImageReadMode.GRAY).type(torch.float64).to(self.device)
                        img1 = read_image(str(self.image_data_cam1_sync[img_indx]),
                                        ImageReadMode.GRAY).type(torch.float64).to(self.device)
                    else:
                        img0 = torch.zeros((1, 480, 752)).to(self.device)
                        img1 = torch.zeros((1, 480, 752)).to(self.device)
                f_B_s[i] = torch.zeros((3, 1)).to(self.device)
                f_W_s[i] = torch.zeros((3, 1)).to(self.device)
            else:
                f_B_s[i] = f_B_s[i].to(self.device)
                f_W_s[i] = f_W_s[i].to(self.device)
                img_indx = self.log_stereo_idx[idx + i]
                if self.include_imgs:
                    img0 = read_image(str(self.image_data_cam0_sync[img_indx]),
                                    ImageReadMode.GRAY).type(torch.float64).to(self.device)
                    img1 = read_image(str(self.image_data_cam1_sync[img_indx]),
                                    ImageReadMode.GRAY).type(torch.float64).to(self.device)
            if self.include_imgs:
                img0_list.append(img0 / 255.00)
                img1_list.append(img1 / 255.00)
        if self.include_imgs:
            img0_list = torch.stack(img0_list)
            img1_list = torch.stack(img1_list)
        else:
            img0_list = xhat_k_1
            img1_list = xhat_k_1
        return img0_list, img1_list, xhat_k_1, P_k_1, f_B_s, f_W_s, imu, x_k, f_n

    
def collate_fn_(batch):
    img0s, img1s, xhat_k_1s, P_k_1s, f_Bs, f_Ws, imus, x_ks, f_ns = zip(*batch)
    # Drop images if they are zero
    batch_size = len(img0s)
    img0s_processed = []
    img1s_processed = []
    f_Bs_processed = []
    f_Ws_processed = []
    max_length_f_Bs = max(max(tensor.size(-1) for tensor in f_B) for f_B in f_Bs)
    max_length_f_Ws = max(max(tensor.size(-1) for tensor in f_W) for f_W in f_Ws)
    for i in range(batch_size):
        img0s_processed.append(torch.stack([img for img in img0s[i] if not \
                                             torch.all(torch.eq(img, 0))]))
        img1s_processed.append(torch.stack([img for img in img1s[i] if not \
                                             torch.all(torch.eq(img, 0))]))
        
        f_Bs_processed.append(torch.stack([torch.cat([tensor, torch.zeros( 3,
                                                               max_length_f_Bs - tensor.size(-1),
                                                               device=tensor.device)], dim=1) 
                                                               for tensor in f_Bs[i]]))
        f_Ws_processed.append(torch.stack([torch.cat([tensor, torch.zeros(3,
                                                                max_length_f_Ws - tensor.size(-1),
                                                                device=tensor.device)], dim=1) 
                                                                for tensor in f_Ws[i]]))

    return (
        torch.stack(img0s), 
        torch.stack(img1s), 
        torch.stack(xhat_k_1s), 
        torch.stack(P_k_1s), 
        torch.stack(f_Bs_processed), 
        torch.stack(f_Ws_processed), 
        torch.stack(imus), 
        torch.stack(x_ks), 
        torch.stack(f_ns)
    )



# def collate_fn_(batch):
#     img0s , img1s , xhat_k_1s, P_k_1s, f_Bs, f_Ws, imus, x_ks , f_ns = zip(*batch)
#     # for i, tensor in enumerate(f_Bs):
#     #     print(f"Tensor {i} shape: {tensor.shape}")
#     # Ensure that tensors are of the same dimension along all but the first dimension
#     # Find the maximum length of the sequences
#     max_length_f_Bs = max(tensor.size(1) for tensor in f_Bs)
#     max_length_f_Ws = max(tensor.size(1) for tensor in f_Ws)

#     # Pad sequences to the maximum length along the second dimension
#     f_Bs_padded = [torch.cat([tensor, torch.zeros(3, max_length_f_Bs - tensor.size(1) ,
#  device=tensor.device)], dim=1) for tensor in f_Bs]
#     f_Ws_padded = [torch.cat([tensor, torch.zeros(3, max_length_f_Ws - tensor.size(1) ,
#  device=tensor.device)], dim=1) for tensor in f_Ws]

#     # Stack the padded sequences
#     f_Bs_stacked = torch.stack(f_Bs_padded)
#     f_Ws_stacked = torch.stack(f_Ws_padded)
    
#     return torch.stack(img0s), torch.stack(img1s), torch.stack(xhat_k_1s), torch.stack(P_k_1s),
#  f_Bs_stacked, f_Ws_stacked, \
#  torch.stack(imus), torch.stack(x_ks) , torch.stack(f_ns)
    # Pad sequences
    # f_Bs_padded = pad_sequence(f_Bs, batch_first=True, padding_value=0)
    # f_Ws_padded = pad_sequence(f_Ws, batch_first=True, padding_value=0)
    
    # return img0s, img1s, xhat_k_1s, P_k_1s, f_Bs_padded, f_Ws_padded, imus, x_ks
    # f_Bs_packed = pack_sequence(f_Bs, enforce_sorted=False)
    # f_Ws_packed = pack_sequence(f_Ws, enforce_sorted=False)
    # f_Bs_stacked = torch.stack(f_Bs_packed)
    # f_Ws_stacked = torch.stack(f_Ws_packed)
    # return img0s , img1s , xhat_k_1s, P_k_1s, f_Bs_stacked, f_Ws_stacked, imus, x_ks
