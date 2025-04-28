import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import glob
import yaml
import os
from numpy.linalg import inv
from matplotlib import pyplot as plt
from pytorch3d.transforms import quaternion_to_matrix
from copy import deepcopy


class EuRoCReader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.cam0_path = self.dataset_path / 'mav0' / 'cam0' / 'data'
        self.cam1_path = self.dataset_path / 'mav0' / 'cam1' / 'data'
        self.imu_path = self.dataset_path / 'mav0' / 'imu0' / 'data.csv'
        self.gt_file = self.dataset_path / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
        self.ground_truth = self.read_ground_truth()

    def read_ground_truth(self):
        ground_truth = pd.read_csv(self.gt_file)
        # ground_truth.columns = ['timestamp', 'p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'b_w_x', 'b_w_y', 'b_w_z', 'b_a_x', 'b_a_y', 'b_a_z']
        ground_truth.columns = ['timestamp', 'p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z', 'v_x', 'v_y', 'v_z', 'b_w_x', 'b_w_y', 'b_w_z', 'b_a_x', 'b_a_y', 'b_a_z']
        ground_truth['timestamp'] = ground_truth['timestamp'].astype(np.int64)
        return ground_truth

    def read_imu_data(self):
        imu_data = pd.read_csv(self.imu_path)
        imu_data.columns = ['timestamp', 'omega_x', 'omega_y', 'omega_z', 'alpha_x', 'alpha_y', 'alpha_z']
        imu_data['timestamp'] = imu_data['timestamp'].astype(np.int64)
        return imu_data

    def read_images(self):
        cam0_images = sorted(self.cam0_path.glob('*.png'))
        cam1_images = sorted(self.cam1_path.glob('*.png'))
        return cam0_images, cam1_images

    def synchronize_data(self, retain_percentage=100):
        imu_data = self.read_imu_data()
        ground_truth = self.read_ground_truth()
        image_data_cam0, image_data_cam1 = self.read_images()

        common_timestamps = set(imu_data['timestamp']).intersection(ground_truth['timestamp']).intersection(
            [int(p.stem) for p in image_data_cam0]).intersection([int(p.stem) for p in image_data_cam1])

        if not common_timestamps:
            raise ValueError("No common timestamps found between the datasets")

        first_common_timestamp = min(common_timestamps)

        imu_data_sync = imu_data[imu_data['timestamp'] >= first_common_timestamp].reset_index(drop=True)
        ground_truth_sync = ground_truth[ground_truth['timestamp'] >= first_common_timestamp].reset_index(drop=True)
        image_data_cam0_sync = [img for img in image_data_cam0 if int(img.stem) >= first_common_timestamp]
        image_data_cam1_sync = [img for img in image_data_cam1 if int(img.stem) >= first_common_timestamp]

        retain_count = int(len(imu_data_sync) * (retain_percentage / 100.0))
        
        self.imu_data_sync = imu_data_sync.iloc[:retain_count]
        self.ground_truth_sync = ground_truth_sync.iloc[:retain_count]
        self.image_data_cam0_sync = image_data_cam0_sync[:retain_count]
        self.image_data_cam1_sync = image_data_cam1_sync[:retain_count]

        # return self.imu_data_sync, self.ground_truth_sync, self.image_data_cam0_sync, self.image_data_cam1_sync


class StereoFeature:
    def __init__(self, K0, D0, K1, D1, R, T):
        self.K0 = K0
        self.D0 = D0
        self.K1 = K1
        self.D1 = D1
        self.R = R
        self.T = T

        # Rectification parameters
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.stereoRectify(K0, D0, K1, D1, (752, 480), R, T)[:5]

        # Rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K0, D0, self.R1, self.P1, (752, 480), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K1, D1, self.R2, self.P2, (752, 480), cv2.CV_32FC1)

    def detect_and_match_features(self, img1, img2):
        # Rectify images
        img1_rect = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        self.left_rectified = img1_rect
        self.right_rectified = img2_rect

        orb = cv2.ORB_create(patchSize=64 , WTA_K = 4 , nlevels = 5 , nfeatures=2000, fastThreshold=10)
        # orb = cv2.ORB_create()
        kp1, self.des1 = orb.detectAndCompute(img1_rect, None)
        kp2, self.des2 = orb.detectAndCompute(img2_rect, None)

        # sift = cv2.SIFT_create()
        # kp1, self.des1 = sift.detectAndCompute(img1_rect, None)
        # kp2, self.des2 = sift.detectAndCompute(img2_rect, None)

        # akaze = cv2.AKAZE_create()

        # # Detect keypoints and compute descriptors
        # kp1, self.des1 = akaze.detectAndCompute(img1_rect, None)
        # kp2, self.des2 = akaze.detectAndCompute(img2_rect, None)


        # original_kp1_map = {kp: idx for idx, kp in enumerate(kp1)}
        # original_kp2_map = {kp: idx for idx, kp in enumerate(kp2)}

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #for orb and akaze
        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) #for sift
        # distance_threshold = 30 
        distance_threshold = np.inf
        matches = bf.match(self.des1, self.des2)
        self.matches = [m for m in matches
                         if ((abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < 2) and
                             (kp1[m.queryIdx].pt[0] >= kp2[m.trainIdx].pt[0]) and (m.distance < distance_threshold))]
        
        # Extract matched points
        pts1= np.float32([kp1[m.queryIdx].pt for m in self.matches])
        self.kp1 = [kp1[m.queryIdx] for m in self.matches]
        # des1 = np.uint8([self.des1[m.queryIdx] for m in self.matches]) #for orb and akaze
        des1 = np.float32([self.des1[m.queryIdx] for m in self.matches]) #for sift
        pts2 = np.float32([kp2[m.trainIdx].pt for m in self.matches])
        self.kp2 = [kp2[m.trainIdx] for m in self.matches]
        # des2 = np.uint8([self.des2[m.trainIdx] for m in self.matches]) #for orb and akaze
        des2 = np.float32([self.des2[m.trainIdx] for m in self.matches]) #for sift

        # Create a mapping from the updated keypoints to their new indices
        kp1_map = {kp: idx for idx, kp in enumerate(self.kp1)}
        kp2_map = {kp: idx for idx, kp in enumerate(self.kp2)}

        # Update the matches to use the new indices
        updated_matches = []
        for m in self.matches:
            if kp1[m.queryIdx] in kp1_map and kp2[m.trainIdx] in kp2_map:
                new_query_idx = kp1_map[kp1[m.queryIdx]]
                new_train_idx = kp2_map[kp2[m.trainIdx]]
                updated_match = cv2.DMatch(new_query_idx, new_train_idx, m.distance)
                updated_matches.append(updated_match)

        # Replace the old matches with the updated matches
        self.matches = updated_matches

        # Outlier rejection using the Fundamental Matrix
        
        # # pts1, pts2, des1, des2 = self.statistical_outlier_removal(pts1, pts2, des1, des2)
        # self.pts1, self.pts2, self.des1, self.des2 , self.kp1 , self.kp2 , self.matches= self.reject_outliers(pts1,
        #                                                                                         pts2,
        #                                                                                           des1,
        #                                                                    des2 , self.kp1 , self.kp2 , self.matches)
        self.pts1, self.pts2, self.des1, self.des2 = pts1, pts2, des1, des2
    def statistical_outlier_removal(self, pts1, pts2, des1, des2):
        def filter_inliers(points):
            points_mean = np.mean(points, axis=0)
            dists = np.sqrt(np.sum((points - points_mean) ** 2, axis=1))
            dist_mean = np.mean(dists)
            dist_std = np.std(dists)
            inliers = np.abs(dists - dist_mean) <= dist_std
            return inliers
        
        # Find inliers for both sets of points
        inliers_pts1 = filter_inliers(pts1)
        inliers_pts2 = filter_inliers(pts2)
        
        # Keep only the common inliers across both sets of points
        common_inliers = inliers_pts1 & inliers_pts2
        
        return pts1[common_inliers], pts2[common_inliers], des1[common_inliers], des2[common_inliers]

    def reject_outliers(self, pts1, pts2, des1, des2, kp1, kp2, matches):
        if pts1.size == 0 or pts2.size == 0:
            return [], [], [], [], [], [], []
        _ , mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3)
        if mask is None:
            return [], [], [], [], [], [], []
        mask = mask.ravel().astype(bool)
        
        # Filter points, descriptors, and keypoints based on the mask
        pts1_inliers = pts1[mask]
        pts2_inliers = pts2[mask]
        des1_inliers = des1[mask]
        des2_inliers = des2[mask]
        kp1_inliers = np.array(kp1)[mask].tolist()
        kp2_inliers = np.array(kp2)[mask].tolist()
        
        # Update matches
        
        kp1_map = {kp: idx for idx, kp in enumerate(kp1_inliers)}
        kp2_map = {kp: idx for idx, kp in enumerate(kp2_inliers)}

        valid_matches = []
        for m in matches:
            if kp1[m.queryIdx] in kp1_map and kp2[m.trainIdx] in kp2_map:
                new_query_idx = kp1_map[kp1[m.queryIdx]]
                new_train_idx = kp2_map[kp2[m.trainIdx]]
                updated_match = cv2.DMatch(new_query_idx, new_train_idx, m.distance)
                valid_matches.append(updated_match)

        
        return pts1_inliers, pts2_inliers, des1_inliers, des2_inliers, kp1_inliers, kp2_inliers, valid_matches

    
    def save_matched_points_plot(self, filename, figsize=(20, 10)):
        try:
            # Add text on the left and right images before drawing matches
            left_image_with_text = self.left_rectified.copy()
            right_image_with_text = self.right_rectified.copy()

            # Add 'Left Image' text on the left image
            cv2.putText(left_image_with_text, 'Left Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Add 'Right Image' text on the right image
            cv2.putText(right_image_with_text, 'Right Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw matches between the images
            img_matches = cv2.drawMatches(
                left_image_with_text, self.kp1,
                right_image_with_text, self.kp2,
                self.matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Use Matplotlib to save with title and annotations
            plt.figure(figsize=figsize)
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            # plt.title('Matched Points')
            plt.axis('off')

            # Save the figure
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error saving matched points plot: {e}")
    
class StereoTrinagulation:
    def __init__(self, StereoFeature , extrinsics_L_inv):
        self.P1 = StereoFeature.P1
        self.P2 = StereoFeature.P2
        self.pts1 = StereoFeature.pts1_
        self.pts2 = StereoFeature.pts2_
        self.cam_to_body_R = extrinsics_L_inv[0:3, 0:3]
        self.cam_to_body_T = extrinsics_L_inv[0:3, 3].reshape((3, 1))

    def triangulate_points(self):
        pts4D = cv2.triangulatePoints(self.P1, self.P2, self.pts1.T, self.pts2.T)
        pts3D = pts4D[:3] / pts4D[3]
        # pts3D *= 0.001 # Convert to m
        return pts3D.T

    def transform_to_body_frame(self):
        pts3D = self.triangulate_points()
        pts3D_body = (self.cam_to_body_R @ pts3D.T + self.cam_to_body_T).T
        return pts3D_body

    def transform_to_world_frame(self, body_to_world_R, body_to_world_T):
        pts3D_body = self.transform_to_body_frame()
        pts3D_world = (body_to_world_R @ pts3D_body.T + body_to_world_T.reshape(-1, 1)).T
        return pts3D_world


def quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z):
    quat_tensor = torch.tensor([q_w, q_x, q_y, q_z], dtype=torch.float32)
    rot_matrix = quaternion_to_matrix(quat_tensor)
    rot_matrix_np = rot_matrix.numpy()
    return rot_matrix_np

class TemporalMatch:
    """
    Class to model matches over time
    """

    def __init__(self, stereo_pair_1, stereo_pair_2):
        self.stereo_pair_1 = stereo_pair_1
        self.stereo_pair_2 = stereo_pair_2

        #  create BFMatcher object
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #for orb and akaze
        bf = cv2.BFMatcher(cv2.NORM_L2) #for sift

        # Match descriptors across time - left image
        # self.matches = bf.match(stereo_pair_1.des1, stereo_pair_2.des1)
        matches = bf.knnMatch(stereo_pair_1.des1, stereo_pair_2.des1, k=2)

        # Apply ratio test
        good_matches = []
        for match in matches:
            if len(match) < 2:
                good_matches.append(match[0])
                continue
            m, n = match
            if m.distance < 0.85 * n.distance: #0.75 to 0.85
                good_matches.append(m)

        # Store the good matches in the class attribute
        self.matches = good_matches
        # distance_threshold = 5
        # distance_threshold = 20
        distance_threshold = 50 #20 to 50
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        self.matches = [m for m in self.matches if m.distance < distance_threshold]
        # Extract matched points
        self.P1_l = np.float32([self.stereo_pair_1.kp1[m.queryIdx].pt for m in self.matches])
        self.P1_kp1 = [self.stereo_pair_1.kp1[m.queryIdx] for m in self.matches]
        self.P1_r = np.float32([self.stereo_pair_1.kp2[m.queryIdx].pt for m in self.matches])
        self.P1_kp2 = [self.stereo_pair_1.kp2[m.queryIdx] for m in self.matches]
        self.P2_l = np.float32([self.stereo_pair_2.kp1[m.trainIdx].pt for m in self.matches])
        self.P2_kp1 = [self.stereo_pair_2.kp1[m.trainIdx] for m in self.matches]
        self.P2_r = np.float32([self.stereo_pair_2.kp2[m.trainIdx].pt for m in self.matches])
        self.P2_kp2 = [self.stereo_pair_2.kp2[m.trainIdx] for m in self.matches]

        # Descriptors
        self.des1 = np.uint8([stereo_pair_1.des1[m.queryIdx] for m in self.matches])
        self.des2 = np.uint8([stereo_pair_2.des1[m.trainIdx] for m in self.matches])

        kpl_1_map = {kp: idx for idx, kp in enumerate(self.P1_kp1)}
        kpl_2_map = {kp: idx for idx, kp in enumerate(self.P2_kp1)}

        valid_matches = []
        for m in self.matches:
            if self.stereo_pair_1.kp1[m.queryIdx] in kpl_1_map and self.stereo_pair_2.kp1[m.trainIdx] in kpl_2_map:
                new_query_idx = kpl_1_map[self.stereo_pair_1.kp1[m.queryIdx]]
                new_train_idx = kpl_2_map[self.stereo_pair_2.kp1[m.trainIdx]]
                updated_match = cv2.DMatch(new_query_idx, new_train_idx, m.distance)
                valid_matches.append(updated_match)
        
        self.matches = valid_matches

        # Reject outliers
        # self.P1_l, self.P1_r, self.P2_l, self.P2_r, self.des1, self.des2 = self.reject_outliers(self.P1_l, self.P1_r, self.P2_l, self.P2_r, self.des1, self.des2)
        # self.P1_l, self.P1_r, self.P2_l, self.P2_r, self.des1, self.des2 = self.statistical_outlier_removal(self.P1_l, self.P1_r, self.P2_l, self.P2_r, self.des1, self.des2)

        mask = self.reject_outliers(self.P1_l, self.P2_l)
        # mask = np.ones(len(self.P1_l), dtype=bool)
        self.P1_l = self.P1_l[mask]
        self.P1_r = self.P1_r[mask]
        self.P2_l = self.P2_l[mask]
        self.P2_r = self.P2_r[mask]
        self.des1 = self.des1[mask]
        self.des2 = self.des2[mask]
        self.P1_kp1_ = [pt for pt in self.P1_kp1]
        self.P2_kp1_ = [pt for pt in self.P2_kp1]
        self.P1_kp1 = np.array(self.P1_kp1)[mask].tolist()
        self.P1_kp2 = np.array(self.P1_kp2)[mask].tolist()
        self.P2_kp1 = np.array(self.P2_kp1)[mask].tolist()
        self.P2_kp2 = np.array(self.P2_kp2)[mask].tolist()
        # Update matches
        
        kpl_1_map = {kp: idx for idx, kp in enumerate(self.P1_kp1)}
        kpl_2_map = {kp: idx for idx, kp in enumerate(self.P2_kp1)}

        valid_matches = []
        for m in self.matches:
            if self.P1_kp1_[m.queryIdx] in kpl_1_map and self.P2_kp1_[m.trainIdx] in kpl_2_map:
                new_query_idx = kpl_1_map[self.P1_kp1_[m.queryIdx]]
                new_train_idx = kpl_2_map[self.P2_kp1_[m.trainIdx]]
                updated_match = cv2.DMatch(new_query_idx, new_train_idx, m.distance)
                valid_matches.append(updated_match)
        
        self.matches = valid_matches
    
    # def reject_outliers(self, P1_l, P1_r, P2_l, P2_r, des1, des2):
    def reject_outliers(self, P1_l, P2_l):
        if P1_l.size == 0 or P2_l.size == 0:
            return np.zeros(P1_l.shape[0], dtype=bool)
        #3 to 5
        F, mask = cv2.findFundamentalMat(P1_l, P2_l, cv2.FM_RANSAC , ransacReprojThreshold = 5)
        # P1_l_inliers = P1_l[mask.ravel() == 1]
        # P1_r_inliers = P1_r[mask.ravel() == 1]
        # P2_l_inliers = P2_l[mask.ravel() == 1]
        # P2_r_inliers = P2_r[mask.ravel() == 1]
        # des1_inliers = des1[mask.ravel() == 1]
        # des2_inliers = des2[mask.ravel() == 1]
        # return P1_l_inliers, P1_r_inliers, P2_l_inliers, P2_r_inliers, des1_inliers, des2_inliers
        if mask is None:
            return np.zeros(P1_l.shape[0], dtype=bool)
        return mask.ravel() == 1
    
    def statistical_outlier_removal(self, P1_l, P1_r, P2_l, P2_r, des1, des2):
        def filter_inliers(points):
            points_mean = np.mean(points, axis=0)
            dists = np.sqrt(np.sum((points - points_mean) ** 2, axis=1))
            dist_mean = np.mean(dists)
            dist_std = np.std(dists)
            inliers = np.abs(dists - dist_mean) <= dist_std
            return inliers
    
        # Find inliers for each set of points
        inliers_P1_l = filter_inliers(P1_l)
        inliers_P1_r = filter_inliers(P1_r)
        inliers_P2_l = filter_inliers(P2_l)
        inliers_P2_r = filter_inliers(P2_r)
    
        # Keep only the common inliers across all sets of points
        common_inliers = inliers_P1_l & inliers_P1_r & inliers_P2_l & inliers_P2_r
        
        return P1_l[common_inliers], P1_r[common_inliers], P2_l[common_inliers], P2_r[common_inliers] , des1[common_inliers] , des2[common_inliers]

    def display_matches(self , save_as):
        img = cv2.drawMatches(self.stereo_pair_1.left_rectified, self.P1_kp1,
                              self.stereo_pair_2.left_rectified, self.P2_kp1,
                              self.matches, None, flags=2)
        cv2.imwrite(save_as, img)
        # plt.imshow(img)
        # plt.show()

    def get_normalized_matches(self, max_matches=np.inf):
        P1_l, P1_r, P2_l, P2_r = self.P1_l, self.P1_r, self.P2_l, self.P2_r
        match_count = len(P1_l)
        if match_count > max_matches:
            P1_l = P1_l[:max_matches]
            P1_r = P1_r[:max_matches]
            P2_l = P2_l[:max_matches]
            P2_r = P2_r[:max_matches]
        
        return P1_l, P1_r, P2_l, P2_r, len(P1_l)




class Calibration:
    """
    Class for all stereo calibration related operations
    """

    def __init__(self, dataset_path):
        """
        Initialization function
        """
        print("Initializing calibration object!")
        self.dataset_path = Path(dataset_path)
        self.left_file = self.dataset_path / 'mav0' / 'cam0' / 'sensor.yaml'
        self.right_file = self.dataset_path / 'mav0' / 'cam1' / 'sensor.yaml'
        self.left_height = 0
        self.left_width = 0
        self.right_height = 0
        self.right_width = 0
        self.left_K = []
        self.left_D = []
        self.right_K = []
        self.right_D = []
        self.extrinsics_R = []
        self.extrinsics_T = []
        self.tr_base_left = []
        self.tr_base_right = []
        self.load_calibration()

    # Parse camera calibration yaml file - intrinsics
    def load_intrinsics(self, calib_data):
        """
        Load EUROC camera intrinsic data
        Taken from: https://github.com/lrse/sptam
        """
        width, height = calib_data['resolution']
        D = np.array(calib_data['distortion_coefficients'])
        fx, fy, px, py = calib_data['intrinsics']
        K = np.array([[fx, 0, px],
                      [0, fy, py],
                      [0, 0, 1]])
        return height, width, K, D

    # Parse camera calibration yaml file - extrinsics
    def load_extrinsics(self, calib_data):
        """
        Load EUROC stereo extrinsics data
        Taken from: https://github.com/lrse/sptam
        """
        # read homogeneous rotation and translation matrix
        transformation_base_camera = np.array(calib_data['T_BS']['data'])
        transformation_base_camera = transformation_base_camera.reshape((4, 4))
        return transformation_base_camera

    # Read calibration file into single structure
    def load_calibration(self):
        """
        Load calibration data into self object
        """
        # Open .yaml files
        left_calib = open(self.left_file, 'r')
        left_calib_data = yaml.load(left_calib, Loader=yaml.FullLoader)
        right_calib = open(self.right_file, 'r')
        right_calib_data = yaml.load(right_calib, Loader=yaml.FullLoader)

        # Parse yaml contents - intrinsics
        self.left_height, self.left_width, self.left_K, self.left_D \
            = self.load_intrinsics(left_calib_data)
        self.right_height, self.right_width, self.right_K, self.right_D \
            = self.load_intrinsics(right_calib_data)

        # Parse yaml contents - extrinsics
        tr_base_left = self.load_extrinsics(left_calib_data)
        tr_base_right = self.load_extrinsics(right_calib_data)

        self.tr_base_left = tr_base_left
        self.tr_base_right = tr_base_right

        # Calculate transformation from L_camera to R_camera
        tr_right_base = inv(tr_base_right)
        tr_right_left = tr_right_base.dot(tr_base_left)
        # Assign extrinsics to class
        self.extrinsics_R = tr_right_left[0:3, 0:3]
        self.extrinsics_T = tr_right_left[0:3, 3]

        tr_left_base = inv(tr_base_left)
        self.tr_left_base = tr_left_base
        tr_left_right = tr_left_base.dot(tr_base_right)
        # self.extrinsics_R = tr_left_right[0:3, 0:3]
        # self.extrinsics_T = tr_left_right[0:3, 3]
        

    # Display camera intrinsics
    def display_intrinsics(self, camera):
        """
        Print camera intrinsics
        """
        if camera == "left":
            print("==== Left Camera ====")
            print("Height : {}".format(self.left_height))
            print("Width : {}".format(self.left_width))
            print("K : {}".format(self.left_K))
            print("D : {}".format(self.left_D))
        elif camera == "right":
            print("==== Right Camera ====")
            print("Height : {}".format(self.right_height))
            print("Width : {}".format(self.right_width))
            print("K : {}".format(self.right_K))
            print("D : {}".format(self.right_D))
        else:
            print("Use option 'left' or 'right' only!")
            exit()

    # Display camera extrinsics
    def display_extrinsics(self):
        """
        Print camera extrinsics
        """
        print("==== Camera Extrinsics ====")
        print("Rotation: {}".format(self.extrinsics_R))
        print("Translation: {}".format(self.extrinsics_T))



# Example usage
if __name__ == "__main__":
    dataset_path = 'V1_02_medium'  # Update this path to your dataset location

    calibration = Calibration(dataset_path)
    print(f"Left camera directory: {calibration.left_file}")
    print(f"Right camera directory: {calibration.right_file}")
    


    euroc_reader = EuRoCReader(dataset_path)
    stereo_triangulation = StereoFeature()

    # process(euroc_reader, stereo_triangulation, K, dist, cam_to_body_R, cam_to_body_T, retain_percentage=20)
