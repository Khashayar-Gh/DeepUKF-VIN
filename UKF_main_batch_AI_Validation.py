import os
os.environ["OMP_NUM_THREADS"] = "20" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "20" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "20" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "20" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "20" # export NUMEXPR_NUM_THREADS=1
import torch
torch.set_num_threads(20)
from dataset_utils import EuRoCReader , TemporalMatch, StereoFeature , Calibration , StereoTrinagulation , quaternion_to_rotation_matrix
from QNUKF import QNUKF
import cv2
import numpy as np
import time
from utils import qMq , datalogger
from copy import deepcopy
from six_dof_VIN import Vis_meas_model
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch3d
import pandas as pd
import pickle
# from EKF_utils import ErrorStateEKF
from AI_data_utils import CustomDataset , collate_fn_
from torch.utils.data import  DataLoader
from AI_model_utils import CNNModel , TimeSeriesGRU , CNNModel3 , TimeSeriesGRU3
torch.cuda.empty_cache()



torch.set_default_dtype(torch.float64)


def process(euroc_reader, stereo_feature, filter,calibration, min_common_points=3 , min_common_points_reset = 6 , max_points = 10):
    # h_n = pd.read_csv('h_n.csv')
    log_obj = datalogger()
    start_time = time.time()

    # with open('log_obj_EKF_20240924233405.pkl', 'rb') as file:
    #     loaded_data = pickle.load(file)
    with open('log_obj_EKF_20241015143753.pkl', 'rb') as file: #for V1_02
        loaded_data = pickle.load(file)

    # with open('log_obj_UKF_20241111165454.pkl', 'rb') as file: #for V2_02
    #     loaded_data = pickle.load(file)
    
    use_AI = True
    print(f"Using AI models {use_AI}")

    xhat_k_1_log ,x_k_log ,P_k_1_log , f_B_log ,f_W_log , imu_log ,imu_noise_log , f_n_log = loaded_data.get()
    filter.x = xhat_k_1_log[0]
    filter.P = P_k_1_log[0]
    EuRoCReader_obj = loaded_data.EuRoCReader
    log_stereo_idx = loaded_data.stereo_idx
    f_n_log = [torch.tensor(f_n_) if f_n_ is not None else torch.tensor(0.0) for f_n_ in f_n_log]
    imu_s = loaded_data.imu_s
    time_s = loaded_data.time
    time_ind = loaded_data.time_index

    device = 'cuda'

    dataset = CustomDataset(xhat_k_1_log ,P_k_1_log , f_B_log ,f_W_log , imu_log, x_k_log ,
                        EuRoCReader_obj , log_stereo_idx , f_n_log , device=device , include_imgs = True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False , collate_fn=collate_fn_)
    dataloader_iter = iter(data_loader)
    # vision_model_path = 'QNUKF_12072024/vision_model_07 22 19:16:40.pth'
    # vision_model_path = 'vision_model_09 28 04:56:32.pth'
    # vision_model_path = 'vision_model_10 06 16:17:59.pth'
    # imu_model_path = 'imu_model_10 06 16:17:59.pth'

    #logfile_AIEKF_Both_10 22 16:48:58_all_cri_0_m3.txt 32hidden, 32 hidden, model3
    vision_model_path = "vision_model_10 22 23:31:06.pth"
    imu_model_path = "imu_model_10 22 23:31:06.pth"

    vis_input_dim = 6  # Number of input features per time step
    vis_hidden_dim = 32  # Number of hidden units in the GRU
    # vis_hidden_dim = 2  # #for debug
    # vis_num_layers = 5  # Number of GRU layers
    vis_num_layers = 2  # #for debug
    vis_out_dim = 1  # output dim
    vis_dropout_prob = 0.5  # Dropout rate
    vision_model = CNNModel3(vis_out_dim, vis_dropout_prob , vis_hidden_dim).cuda()
    vision_model.load_state_dict(torch.load(vision_model_path , map_location=torch.device(device)))
    vision_model.eval()

    imu_input_dim = 6  # Number of input features per time step
    imu_hidden_dim = 32  # Number of hidden units in the GRU
    # imu_hidden_dim = 2  # #for debug
    # imu_num_layers = 5  # Number of GRU layers
    imu_num_layers = 2  # #for debug
    imu_output_dim = 12  # output dim
    imu_dropout = 0.5  # Dropout rate
    imu_model = TimeSeriesGRU3(imu_input_dim, imu_hidden_dim, imu_num_layers,
                               imu_output_dim , imu_dropout).cuda()
    imu_model.load_state_dict(torch.load(imu_model_path ,
                                          map_location=torch.device(device)))
    imu_model.eval()
    # cam_to_body_R = calibration.tr_base_left[:3 , :3]
    # cam_to_body_T = calibration.tr_base_left[:3, 3].reshape((3, 1))
    start_time = time.time()
    imu_data_sync, ground_truth_sync, image_data_cam0_sync, image_data_cam1_sync = euroc_reader.imu_data_sync , euroc_reader.ground_truth_sync, euroc_reader.image_data_cam0_sync, euroc_reader.image_data_cam1_sync
    omega_mean = torch.tensor(imu_data_sync[['omega_x', 'omega_y', 'omega_z']].mean().values)
    a_mean = torch.tensor(imu_data_sync[['alpha_x', 'alpha_y', 'alpha_z']].mean().values)
    b_w_mean = torch.tensor(ground_truth_sync[['b_w_x', 'b_w_y', 'b_w_z']].mean().values)
    b_a_mean = torch.tensor(ground_truth_sync[['b_a_x', 'b_a_y', 'b_a_z']].mean().values)
    # Creating a dictionary to store images for fast access by timestamp
    cam0_images_dict = {int(img.stem): img for img in image_data_cam0_sync}
    cam1_images_dict = {int(img.stem): img for img in image_data_cam1_sync}

    reference_img0_path, reference_img1_path = image_data_cam0_sync[0], image_data_cam1_sync[0]
    reference_timestamp = int(reference_img0_path.stem)
    gt_ref_pose = ground_truth_sync.iloc[0]

    reference_img0 = cv2.imread(str(reference_img0_path), cv2.IMREAD_GRAYSCALE)
    reference_img1 = cv2.imread(str(reference_img1_path), cv2.IMREAD_GRAYSCALE)
    ref_feature = deepcopy(stereo_feature)
    ref_feature.detect_and_match_features(reference_img0, reference_img1)
    q_w, q_x, q_y, q_z = gt_ref_pose[['q_w', 'q_x', 'q_y', 'q_z']].values
    body_to_world_R = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
    body_to_world_T = gt_ref_pose[['p_x', 'p_y', 'p_z']].values.reshape((3, 1))
    

    # all_timestamps = sorted(set(imu_data_sync['timestamp']).union(set(ground_truth_sync['timestamp'])).union(set(cam0_images_dict.keys())))
    all_timestamps = sorted(set(imu_data_sync['timestamp']))
    # all_timestamps = set(imu_data_sync['timestamp'])
    timestep_count = len(all_timestamps)
    # xhat_log = torch.zeros((timestep_count, 16))
    # xhat_log[0] = filter.x
    xhat_log = filter.x.unsqueeze(0)
    timestamp_log = torch.tensor([all_timestamps[0]] , device=filter.x.device)
    ind_log = [0]
    stereo_index = 1
    error_vectors = []
    zero_match_count = 0
    tracking_imgs_path = 'tracking_imgs'
    os.makedirs(tracking_imgs_path, exist_ok=True)
    log_obj.setEuRoCReader(euroc_reader)
    omega_m_s = torch.empty(0 , 3 , device=filter.x.device)
    a_m_s = torch.empty(0 , 3 , device=filter.x.device)
    for timestep_indx , timestamp in enumerate(all_timestamps[:-1]):
        
        
        percentage_completed = ((timestep_indx + 1) / timestep_count) * 100
        if abs(percentage_completed % 5) < 0.05 or abs(percentage_completed % 5 - 5) < 0.01 or timestep_indx == timestep_count - 1:
            elapsed_time = time.time() - start_time
            # remaining_iterations = timestep_count - (timestep_indx + 1)
            estimated_total_time = (elapsed_time / (timestep_indx + 1)) * timestep_count
            estimated_time_left = estimated_total_time - elapsed_time
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Estimated Time Left: {estimated_time_left:.2f} seconds")
            print(f"Percentage Completed: {percentage_completed:.0f}%\n")
            print(f"quaternion error: {torch.norm(qMq(filter.x[:4] , torch.tensor(ground_truth_sync.iloc[timestep_indx][['q_w', 'q_x', 'q_y', 'q_z']].values)))}")
            print(f"Position error: {torch.norm(filter.x[4:7] - torch.tensor(ground_truth_sync.iloc[timestep_indx][['p_x', 'p_y', 'p_z']].values))}")
            print(f"Velocity error: {torch.norm(filter.x[7:10] - torch.tensor(ground_truth_sync.iloc[timestep_indx][['v_x', 'v_y', 'v_z']].values))}")
        
        


        imu_data = imu_data_sync.iloc[timestep_indx]
        omega_m = torch.tensor(imu_data[['omega_x', 'omega_y', 'omega_z']].values).unsqueeze(0)
        a_m = torch.tensor(imu_data[['alpha_x', 'alpha_y', 'alpha_z']].values).unsqueeze(0)
        gt_pose = ground_truth_sync.iloc[timestep_indx+1]
        q_w, q_x, q_y, q_z = gt_pose[['q_w', 'q_x', 'q_y', 'q_z']].values
        gt_R = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
        gt_T = gt_pose[['p_x', 'p_y', 'p_z']].values.reshape((3, 1))
        if timestep_indx == 39:
            print(f"gt position: {gt_T}")
            print(f"gt quaternion: {gt_pose[['q_w', 'q_x', 'q_y', 'q_z']].values}")
            print(f"gt velocity: {gt_pose[['v_x', 'v_y', 'v_z']].values}")
            pass
        assert abs(timestamp - int(ground_truth_sync.iloc[timestep_indx]['timestamp']))<1/200*1e9, "IMU and ground truth timestamps are not synchronized"

        omega_mean = torch.tensor([0.1356,0.0386,0.0242])# for debugging
        # omega_std = omega_mean*1e-2
        omega_std = omega_mean*1e-3
        a_mean = torch.tensor([9.2501 ,0.0293,-3.3677]) # for debugging
        a_std = a_mean*1e-4
        b_w_mean = torch.tensor([-0.0022, 0.0208 , 0.0758]) # for debugging
        # omega_bias_std = b_w_mean*1e-4
        omega_bias_std = b_w_mean*1e-1
        b_a_mean = torch.tensor([-0.0147, 0.1051, 0.0930]) # for debugging
        a_bias_std = b_a_mean*1e-2
        
        # omega_std , a_std , omega_bias_std , a_bias_std = omega_std*1e-1 , a_std*1e-1 , omega_bias_std*1e-1 , a_bias_std*1e-1
        
        log_obj.log_stereo_idx(stereo_index)
        # filter.predict(omega_m, a_m, omega_std , a_std , omega_bias_std , a_bias_std)
        # print(qnukfilter.x) # for debugging
        # print(qnukfilter.P) # for debugging
        pts3D_body = None
        pts3D_world = None
        imu_noise_log_its = iter(imu_noise_log)
        
        
        if all_timestamps[timestep_indx+1] >= int(image_data_cam0_sync[stereo_index].stem):

            assert abs(all_timestamps[timestep_indx+1] - int(image_data_cam0_sync[stereo_index].stem))<1/200*1e9 , "camera IMU timestamps are not synchronized"

            img0 = cv2.imread(image_data_cam0_sync[stereo_index], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(image_data_cam1_sync[stereo_index], cv2.IMREAD_GRAYSCALE)
            cur_feature = deepcopy(stereo_feature)
            cur_feature.detect_and_match_features(img0, img1)
            # cur_feature.save_matched_points_plot("stereo "+str(timestamp)+".png")
            # Limit the number of points to max_points
            if len(ref_feature.matches) == 0 or len(cur_feature.matches) == 0:
                match_count = 0
            else:
                match_obj = TemporalMatch(ref_feature, cur_feature)
                ref_feature.pts1_, ref_feature.pts2_ , cur_feature.pts1_ , cur_feature.pts2_ , match_count = match_obj.get_normalized_matches(max_points)
            # save_as = os.path.join(tracking_imgs_path, str(timestamp) + "_matches.png")
            # match_obj.display_matches(save_as)
            if match_count >= min_common_points:

                if not gt_ref_pose.empty:
                    ref_triangulation = StereoTrinagulation(ref_feature , calibration.tr_left_base) #tr_left_base is the inverse of the extrinsics of the left camera
                    cur_triangulation = StereoTrinagulation(cur_feature , calibration.tr_left_base)
                    
                    # pts_std = 0.1503928471980654*1
                    # pts_std = 0.17498906758461455
                    # pts_std = 0.02
                    pts_std = 0.3

                    
                    pts3D_body = cur_triangulation.transform_to_body_frame()

                    pts3D_world = ref_triangulation.transform_to_world_frame(body_to_world_R,
                                                                              body_to_world_T)
                    
                    pts3D_body = torch.tensor(pts3D_body).squeeze().t()
                    pts3D_world = torch.tensor(pts3D_world).squeeze().t()
                    f_e_ = torch.flatten(pts3D_body.T) - Vis_meas_model(
                        torch.tensor(gt_pose[['q_w', 'q_x', 'q_y', 'q_z' ,
                                                                 'p_x', 'p_y', 'p_z']].values) ,
                                                                   pts3D_world)
                    error_vectors.extend ([i for i in f_e_.cpu().detach().numpy().flatten()])



                    # print(f"IMU data shape: {torch.cat([omega_m_s , a_m_s] , dim = 1).shape}") # for debugging

                    img0_data_loaded , img1_data_loaded , _ , _ , \
                        _, _, imu_data_loaded, _ , \
                            _ = next(dataloader_iter)
                    del _
                    img0 = img0_data_loaded[: , -1 ,...]
                    img1 = img1_data_loaded[: , -1 ,...]
                    imu_data_loaded = imu_data_loaded[: , -1 , -10: ,...]
                    # print(f"Image data shape: {img0_data_loaded.shape}")
                    # print(f"IMU data shape: {imu_data_loaded.shape}")
                    with torch.no_grad():
                        if use_AI:
                            vision_adaptive_gain = vision_model(img0 , img1)
                            imu_adaptive_gain = imu_model(imu_data_loaded)
                        else:
                            imu_adaptive_gain = torch.zeros(6 , device=imu_data_loaded.device)
                            vision_adaptive_gain = torch.tensor(0.0 , device=img0.device)
                        
                        imu_adaptive_gain = imu_adaptive_gain.to('cpu')
                        imu_nominal_std = next(imu_noise_log_its)
                        beta = 1
                        tanh = torch.nn.Tanh()
                        # print(f"imu_adaptive_gain: {imu_adaptive_gain.shape}")
                        imu_adaptive_std = imu_nominal_std*10**(beta*tanh(imu_adaptive_gain[0]))
                        
                        # vision_adaptive_gain = torch.tensor(0.0 , device=img0.device)
                    del img0 , img1 , imu_data_loaded
                    torch.cuda.empty_cache()
                    vision_adaptive_std = pts_std*10**(beta*tanh(vision_adaptive_gain))
                    vision_adaptive_std = vision_adaptive_std.to('cpu')
                    a_m_s = torch.concat([a_m_s , a_m] , dim = 0)
                    omega_m_s = torch.concat([omega_m_s , omega_m] , dim = 0)
                    log_obj.log_P_k_1(filter.P)
                    log_obj.log_xhat_k_1(filter.x)
                    log_obj.log_f_B(pts3D_body)
                    log_obj.log_f_W(pts3D_world)
                    log_obj.log_f_n(pts_std)
                    log_obj.log_imu_noise(imu_nominal_std)
                    log_obj.log_imu_s(torch.cat([omega_m_s , a_m_s] , dim = 1))
                    filter.batch_predict(omega_m_s, a_m_s, imu_adaptive_std[:3] , imu_adaptive_std[3:6] ,
                            imu_adaptive_std[6:9] , imu_adaptive_std[9:12])
                    filter.update(pts3D_world , pts3D_body , vision_adaptive_std)
                    omega_m_s = torch.empty(0 , 3 , device=omega_m.device)
                    a_m_s = torch.empty(0 , 3 , device=a_m.device)
                    # xhat_log[timestep_indx+1] = filter.x
                    xhat_log = torch.cat([xhat_log , filter.x.unsqueeze(0)] , dim = 0)
                    timestamp_log = torch.cat([timestamp_log , torch.tensor([timestamp])])
                    ind_log.append(timestep_indx)
                    log_obj.log_xhat_k(filter.x)
                    log_obj.log_x_k(torch.tensor(gt_pose[['q_w', 'q_x', 'q_y', 'q_z' ,
                                             'p_x', 'p_y', 'p_z' , 
                                             'v_x', 'v_y', 'v_z' , 
                                             'b_w_x', 'b_w_y', 'b_w_z',
                                             'b_a_x', 'b_a_y', 'b_a_z']].values))
                    
                else:
                    # print("No stereo data")
                    # log_obj.log_f_B(None)
                    # log_obj.log_f_W(None)
                    # log_obj.log_f_n(None)
                    a_m_s = torch.concat([a_m_s , a_m] , dim = 0)
                    omega_m_s = torch.concat([omega_m_s , omega_m] , dim = 0)
            else:
                # print("No stereo data")
                # log_obj.log_f_B(None)
                # log_obj.log_f_W(None)
                # log_obj.log_f_n(None)
                a_m_s = torch.concat([a_m_s , a_m] , dim = 0)
                omega_m_s = torch.concat([omega_m_s , omega_m] , dim = 0)
            # print(f"Match count: {match_count}")
            if match_count == 0:
                zero_match_count += 1
            # if match_count == 0 or match_count < min_common_points_reset or stereo_index == 100:
            #     print("Resetting reference image")
            #     pass
            if match_count < min_common_points_reset:
                reference_img0_path, reference_img1_path = image_data_cam0_sync[stereo_index], image_data_cam1_sync[stereo_index]
                # reference_timestamp = int(reference_img0_path.stem)
                gt_ref_pose = ground_truth_sync.iloc[timestep_indx+1]
                reference_img0 = cv2.imread(str(reference_img0_path), cv2.IMREAD_GRAYSCALE)
                reference_img1 = cv2.imread(str(reference_img1_path), cv2.IMREAD_GRAYSCALE)
                ref_feature = deepcopy(stereo_feature)
                ref_feature.detect_and_match_features(reference_img0, reference_img1)
                q_w, q_x, q_y, q_z = gt_ref_pose[['q_w', 'q_x', 'q_y', 'q_z']].values
                body_to_world_R = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
                body_to_world_T = gt_ref_pose[['p_x', 'p_y', 'p_z']].values.reshape((3, 1))


            stereo_index += 1
        else:
            # print("No stereo data")
            # log_obj.log_f_B(None)
            # log_obj.log_f_W(None)
            # log_obj.log_f_n(None)
            a_m_s = torch.concat([a_m_s , a_m] , dim = 0)
            omega_m_s = torch.concat([omega_m_s , omega_m] , dim = 0)
        
        
        # Concatenate all error vectors into a single flat array
    # all_errors = np.concatenate(error_vectors)
    all_errors = np.array(error_vectors)
    # Compute the standard deviation of the error vectors
    std_error = np.std(all_errors , ddof=1) # ddof=1 for unbiased estimation
    mean_error = np.mean(all_errors)
    print(f"Zero match count: {zero_match_count}")
    # Plot the histogram of all_errors
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=30, edgecolor='k', alpha=0.7)
    plt.title('Distribution of All Errors')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_error:.2f}')
    plt.axvline(mean_error + std_error, color='g', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_error:.2f}')
    plt.axvline(mean_error - std_error, color='g', linestyle='dashed', linewidth=1)
    plt.legend()

    # Save the plot
    plt.savefig('all_errors_distribution.png')
    plt.close()

    

    # Print the estimated standard deviation
    print(f'Estimated standard deviation of the error: {std_error}')
    print(f'Estimated mean of the error: {mean_error}')
    print(f"IMU data shape: {imu_data_sync.shape}")
    print(f"Ground truth shape: {ground_truth_sync.shape}")
    print(f"Image data cam0 shape: {len(image_data_cam0_sync)}")
    return ind_log , timestamp_log, xhat_log , log_obj

# Example usage
if __name__ == "__main__":
    # dataset_path = 'V2_02_medium'
    dataset_path = 'V1_02_medium'
    calibration = Calibration(dataset_path)

    euroc_reader = EuRoCReader(dataset_path)
    # euroc_reader.synchronize_data(retain_percentage=98)
    euroc_reader.synchronize_data(retain_percentage=90)
    stereo_feature = StereoFeature(calibration.left_K, calibration.left_D, calibration.right_K, calibration.right_D, calibration.extrinsics_R, calibration.extrinsics_T)
    p0  = torch.tensor(euroc_reader.ground_truth_sync.iloc[0][['p_x', 'p_y', 'p_z']].values)
    q0  = torch.tensor(euroc_reader.ground_truth_sync.iloc[0][['q_w', 'q_x', 'q_y', 'q_z']].values)
    v0  = torch.tensor(euroc_reader.ground_truth_sync.iloc[0][['v_x', 'v_y', 'v_z']].values)
    ba0 = torch.tensor(euroc_reader.ground_truth_sync.iloc[0][['b_a_x', 'b_a_y', 'b_a_z']].values)
    bw0 = torch.tensor(euroc_reader.ground_truth_sync.iloc[0][['b_w_x', 'b_w_y', 'b_w_z']].values)
    bw_mean = torch.tensor(euroc_reader.ground_truth_sync[['b_w_x', 'b_w_y', 'b_w_z']].mean().values)
    ba_mean = torch.tensor(euroc_reader.ground_truth_sync[['b_a_x', 'b_a_y', 'b_a_z']].mean().values)
    # x0 = torch.cat([q0, p0, v0, bw0, ba0])
    x0 = torch.tensor([0.4643,0.7122, -0.4656,0.2458,2.2730,4.4951 ,
                       -1.5434,-0.0034,-0.0106,-0.0055,-0.0022 ,0.0208 ,
                        0.0758,-0.0147, 0.1051, 0.0930])*1.1
    
    # P0 = torch.block_diag(torch.eye(3)*5e3 , torch.eye(3)*6e2 , torch.eye(3)*1e1 ,
    #                        torch.eye(3)*1e1 , torch.eye(3)*1e1)
    P0 = torch.block_diag(torch.eye(3)*7e2 , torch.eye(3)*6e2 , torch.eye(3)*1e1 ,
                           torch.eye(3)*0e1 , torch.eye(3)*0e1)
    # ekffilter = ErrorStateEKF(x0 , P0)
    filter = QNUKF(x0 , P0)
    ind_log , timestamp_log , xhat_log , log_obj = process(euroc_reader, stereo_feature, filter , calibration)
    log_obj.get()
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = f"log_obj_UKF_{timestamp}.pkl"
    print(f"Saving log object to {filename}")

    with open(filename, 'wb') as f:
        pickle.dump(log_obj, f)

    # Plotting part
    time = euroc_reader.ground_truth_sync['timestamp'].values[ind_log]
    time = (time - time[0]) / 1e9
    quaternion = xhat_log[:, :4].numpy()
    position = xhat_log[:, 4:7].numpy()
    velocity = xhat_log[:, 7:10].numpy()

    # Ground truth data
    gt_quaternion = euroc_reader.ground_truth_sync[['q_w', 'q_x', 'q_y', 'q_z']].values[ind_log]
    gt_quaternion = torch.tensor(gt_quaternion)
    gt_quaternion = pytorch3d.transforms.quaternion_to_matrix(gt_quaternion)
    gt_quaternion = pytorch3d.transforms.matrix_to_quaternion(gt_quaternion)
    gt_quaternion = torch.nn.functional.normalize(gt_quaternion, p=2, dim=1)
    gt_quaternion = pytorch3d.transforms.standardize_quaternion(gt_quaternion)
    
    gt_quaternion = np.array(gt_quaternion)
    gt_position = euroc_reader.ground_truth_sync[['p_x', 'p_y', 'p_z']].values[ind_log]
    gt_velocity = euroc_reader.ground_truth_sync[['v_x', 'v_y', 'v_z']].values[ind_log]

    fig, axs = plt.subplots(10, 1, figsize=(10, 20))

    # Plot quaternion components
    for i in range(4):
        axs[i].plot(time, quaternion[:, i], label='Estimated')
        axs[i].plot(time, gt_quaternion[:, i], label='Ground Truth', linestyle='dashed')
        axs[i].set_title(f'Quaternion component q{i}')
        axs[i].legend()

    # Plot position components
    for i in range(3):
        axs[i + 4].plot(time, position[:, i], label='Estimated')
        axs[i + 4].plot(time, gt_position[:, i], label='Ground Truth', linestyle='dashed')
        axs[i + 4].set_title(f'Position component p{i}')
        axs[i + 4].legend()

    # Plot velocity components
    for i in range(3):
        axs[i + 7].plot(time, velocity[:, i], label='Estimated')
        axs[i + 7].plot(time, gt_velocity[:, i], label='Ground Truth', linestyle='dashed')
        axs[i + 7].set_title(f'Velocity component v{i}')
        axs[i + 7].legend()

    plt.tight_layout()
    fig.savefig('z_UKF_AI.png')

    # Calculate error norms
    error_vectors = qMq(torch.tensor(quaternion), torch.tensor(gt_quaternion)).detach().cpu().numpy()
    quaternion_error = np.linalg.norm(error_vectors, axis=1)
    position_error = np.linalg.norm(position - gt_position, axis=1)
    velocity_error = np.linalg.norm(velocity - gt_velocity, axis=1)

    # Plot error norms
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    axs[0].plot(time, quaternion_error, label='Quaternion Error')
    axs[0].set_title('Quaternion Error Norm')
    axs[0].legend()
    axs[1].plot(time, position_error, label='Position Error')
    axs[1].set_title('Position Error Norm')
    axs[1].legend()
    axs[2].plot(time, velocity_error, label='Velocity Error')
    axs[2].set_title('Velocity Error Norm')
    axs[2].legend()
    plt.tight_layout()
    fig.savefig('z_UKF_AI_error.png')
    
    # Plot quaternion components
    # fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    # for i in range(4):
    #     axs[i].plot(time, quaternion[:, i], label='Estimated')
    #     axs[i].plot(time, gt_quaternion[:, i], label='Ground Truth', linestyle='dashed')
    #     axs[i].set_title(f'Quaternion component q{i}')
    #     axs[i].legend()
    # plt.tight_layout()
    # fig.savefig('quaternion_components_comparison.png')

    # # Plot position components
    # fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    # for i in range(3):
    #     axs[i].plot(time, position[:, i], label='Estimated')
    #     axs[i].plot(time, gt_position[:, i], label='Ground Truth', linestyle='dashed')
    #     axs[i].set_title(f'Position component p{i}')
    #     axs[i].legend()
    # plt.tight_layout()
    # fig.savefig('position_components_comparison.png')

    # # Plot velocity components
    # fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    # for i in range(3):
    #     axs[i].plot(time, velocity[:, i], label='Estimated')
    #     axs[i].plot(time, gt_velocity[:, i], label='Ground Truth', linestyle='dashed')
    #     axs[i].set_title(f'Velocity component v{i}')
    #     axs[i].legend()
    # plt.tight_layout()
    # fig.savefig('velocity_components_comparison.png')

    print("Done")