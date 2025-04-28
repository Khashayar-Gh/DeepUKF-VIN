# import pytorch3d
# import pytorch3d.transforms
from pytorch3d.transforms import quaternion_multiply , quaternion_invert, quaternion_to_matrix, standardize_quaternion
from torch.nn.functional import normalize
import torch
import pytorch3d

def qPq(q1 , q2):
    q_sum = normalize(quaternion_multiply(q1, q2), p=2, dim=0)
    q_sum = quaternion_to_matrix(q_sum)
    q_sum = matrix_to_quaternion(q_sum)
    q_sum = normalize(q_sum, p=2, dim=0)
    q_sum = standardize_quaternion(q_sum)
    return q_sum

def qPr(q , r):
    r = normalize_tensor_to_pi(r)
    return qPq(normalize(
        standardize_quaternion(axis_angle_to_quaternion(r))
        , p=2, dim=0) ,
                q)

def qMr(q , r):
    r = normalize_tensor_to_pi(r)
    return qPq(normalize(
        quaternion_invert(standardize_quaternion(axis_angle_to_quaternion(r)))
        , p=2, dim=0) ,
                q)

def qMq(q1 , q2):

    return normalize_tensor_to_pi(quaternion_to_axis_angle(qPq(q1 , quaternion_invert(q2))))

def QWA(quaternions, weights):
    """
    Compute the weighted average of quaternions.
    
    Parameters:
    quaternions (torch.Tensor): A tensor of quaternions with shape (N, 4), where N is the number of quaternions.
    weights (torch.Tensor): A tensor of weights with shape (N).
    
    Returns:
    torch.Tensor: A tensor representing the weighted average quaternion with shape (4).
    """
    # Compute the weighted sum of the quaternions
    M_ = torch.zeros(4, 4 , dtype=quaternions.dtype, device=quaternions.device)
    quaternions = [standardize_quaternion(q) for q in quaternions]
    for i in range(len(quaternions)):    
        M_ += weights[i] * torch.outer(quaternions[i], quaternions[i])
    M_ /= torch.sum(weights)
    try:
        L, V = torch.linalg.eigh(M_)
    except:
        L = torch.tensor([1])
        V = torch.eye(4)
        print('----------------------------------------------------------------')
        print("Error in QWA and DELETE THIS LINE")
    # max_index = torch.argmax(torch.real(L)) # for torch.linalg.eig
    max_index = -1 # for torch.linalg.eigh
    assert torch.any(torch.real(L)>0) , "All eigenvalues are non-positive."
    q_avg = V[:, max_index].real
    q_avg = normalize(q_avg, p=2, dim=0)
    q_avg = standardize_quaternion(q_avg)
    q_avg = quaternion_to_matrix(q_avg)
    q_avg = matrix_to_quaternion(q_avg)
    q_avg = normalize(q_avg, p=2, dim=0)
    q_avg = standardize_quaternion(q_avg)
    return q_avg

def symmetrizeCovariance(P):
    """
    Symmetrize a covariance matrix.
    
    Parameters:
    P (torch.Tensor): A tensor representing the covariance matrix with shape (n, n).
    
    Returns:
    torch.Tensor: A tensor representing the symmetrized covariance matrix with shape (n, n).
    """
    return .5 * (P + P.t())

def normalize_tensor_to_pi(tensor):
    
    norm = torch.norm(tensor)
    if norm.eq(0):
        return tensor  # Return the tensor as is if the norm is zero
    
    # Bring the norm within the range [-pi, pi] using modulo operation
    norm = (norm + torch.pi) % (2 * torch.pi) - torch.pi

    return tensor*norm/torch.norm(tensor)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    #wrapper for pytorch3d.transforms.matrix_to_quaternion
    return pytorch3d.transforms.matrix_to_quaternion(matrix)
    # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    #     raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    # batch_dim = matrix.shape[:-2]
    # m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
    #     matrix.reshape(batch_dim + (9,)), dim=-1
    # )

    # trace = m00 + m11 + m22
    # w, x, y, z = torch.zeros_like(trace), torch.zeros_like(trace), torch.zeros_like(trace), torch.zeros_like(trace)

    # # Case 1: trace > 0
    # t1 = trace > 0
    # s1 = torch.sqrt(1.0 + trace[t1]) * 2
    # w[t1] = 0.25 * s1
    # x[t1] = (m21[t1] - m12[t1]) / s1
    # y[t1] = (m02[t1] - m20[t1]) / s1
    # z[t1] = (m10[t1] - m01[t1]) / s1

    # # Case 2: (R00 > R11) and (R00 > R22)
    # t2 = ~t1 & (m00 > m11) & (m00 > m22)
    # s2 = torch.sqrt(1.0 + m00[t2] - m11[t2] - m22[t2]) * 2
    # w[t2] = (m21[t2] - m12[t2]) / s2
    # x[t2] = 0.25 * s2
    # y[t2] = (m01[t2] + m10[t2]) / s2
    # z[t2] = (m02[t2] + m20[t2]) / s2

    # # Case 3: (R11 > R22)
    # t3 = ~t1 & ~t2 & (m11 > m22)
    # s3 = torch.sqrt(1.0 + m11[t3] - m00[t3] - m22[t3]) * 2
    # w[t3] = (m02[t3] - m20[t3]) / s3
    # x[t3] = (m01[t3] + m10[t3]) / s3
    # y[t3] = 0.25 * s3
    # z[t3] = (m12[t3] + m21[t3]) / s3

    # # Case 4: Otherwise
    # t4 = ~t1 & ~t2 & ~t3
    # s4 = torch.sqrt(1.0 + m22[t4] - m00[t4] - m11[t4]) * 2
    # w[t4] = (m10[t4] - m01[t4]) / s4
    # x[t4] = (m02[t4] + m20[t4]) / s4
    # y[t4] = (m12[t4] + m21[t4]) / s4
    # z[t4] = 0.25 * s4

    # return torch.stack((w, x, y, z), dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    #wrapper for pytorch3d.transforms.quaternion_to_axis_angle
    return pytorch3d.transforms.quaternion_to_axis_angle(quaternions)
    # # Compute the norm of the imaginary part (i, j, k)
    # norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    
    # # Compute the half angles
    # half_angles = torch.atan2(norms, quaternions[..., :1])
    
    # # Compute the full angles
    # angles = 2 * half_angles
    
    # # Handle zero-angle case
    # zero_angles = (norms == 0).expand_as(quaternions[..., 1:])
    
    # # Compute sin(half_angles) / angles for non-zero angles
    # sin_half_angles_over_angles = torch.sin(half_angles) / angles
    
    # # Avoid division by zero for zero angles
    # sin_half_angles_over_angles = torch.where(
    #     zero_angles,
    #     torch.ones_like(sin_half_angles_over_angles),
    #     sin_half_angles_over_angles
    # )
    
    # # Compute the axis-angle representation
    # axis_angle = quaternions[..., 1:] / sin_half_angles_over_angles
    
    # # For zero angles, return the axis as zero vector
    # axis_angle = torch.where(zero_angles, torch.zeros_like(axis_angle), axis_angle)
    
    # return axis_angle


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    #wrapper for pytorch3d.transforms.axis_angle_to_quaternion
    return pytorch3d.transforms.axis_angle_to_quaternion(axis_angle)
    # # Compute the magnitude of the axis_angle vectors to get the angles
    # angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    
    # # Compute the half angles
    # half_angles = angles * 0.5
    
    # # Handle zero-angle case
    # zero_angles = (angles == 0)
    
    # # Compute sin(half_angles) / angles for non-zero angles
    # sin_half_angles_over_angles = torch.sin(half_angles) / angles
    
    # # Avoid division by zero for zero angles
    # sin_half_angles_over_angles = torch.where(
    #     zero_angles.expand_as(axis_angle),
    #     torch.ones_like(sin_half_angles_over_angles),
    #     sin_half_angles_over_angles
    # )
    
    # # Compute the quaternion components
    # quaternions = torch.cat(
    #     [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    # )
    
    # # For zero angles, set the quaternion to (1, 0, 0, 0)
    # quaternions = torch.where(
    #     zero_angles.expand_as(quaternions),
    #     torch.tensor([1.0, 0.0, 0.0, 0.0], device=quaternions.device, dtype=quaternions.dtype).expand_as(quaternions),
    #     quaternions
    # )
    
    # return quaternions

class datalogger():
    def __init__(self):
        self.xhat_k_1 = [] # xhat_{k_1}
        self.xhat_k = [] # xhat_k
        self.x_k = [] # x_k
        self.P_k_1 = [] # P_{k_1}
        self.f_B = [] # feature points in Body frame
        self.f_W = [] # feature points in World frame
        self.imu = [] # measured acceleration angular velocity
        self.imu_s = [] # batched measured acceleration angular velocity
        self.imu_noise = []
        self.f_n = [] # feature points noise
        self.EuRoCReader = None
        self.stereo_idx = []
        self.time_index = []
        self.time = []

    def log_xhat_k(self , data):
        self.xhat_k.append(data)
    def log_xhat_k_1(self , data):
        self.xhat_k_1.append(data)
    def log_x_k(self , data):
        self.x_k.append(data)
    def log_P_k_1(self , data):
        self.P_k_1.append(data)
    def log_f_B(self , data):
        self.f_B.append(data)
    def log_f_W(self , data):
        self.f_W.append(data)
    def log_imu(self , data): # [gyro , acc] #6x1
        """
        data: torch.tensor([gyro_x , gyro_y , gyro_z , acc_x , acc_y , acc_z])
        """
        self.imu.append(data)
    def log_imu_s(self , data): # [gyro , acc] #6x1
        """
        data: N x 6 torch.tensor([gyro_x , gyro_y , gyro_z , acc_x , acc_y , acc_z])
        """
        self.imu_s.append(data)
    def log_imu_noise(self , data):
        """
        data: torch.tensor([gyro_noise_x , gyro_noise_y , gyro_noise_z ,
          acc_noise_x , acc_noise_y , acc_noise_z ,
            gyro_bias_x , gyro_bias_y , gyro_bias_z ,
              acc_bias_x , acc_bias_y , acc_bias_z])
        """
        self.imu_noise.append(data) # [gyro_noise , acc_noise , gyro_bias , acc_bias] #12x1
    def log_f_n(self , data):
        self.f_n.append(data)
    def setEuRoCReader(self , EuRoCReader):
        self.EuRoCReader = EuRoCReader
    def log_stereo_idx(self , data):
        self.stereo_idx.append(data)
    def log_time_index(self , data):
        self.time_index.append(data)
    def log_time(self , data):
        self.time.append(data)
    def get(self):
        # assert len(self.xhat_k_1) == len(self.x_k) == len(self.P_k_1) == len(self.f_B) == len(self.f_W) == len(self.imu) == len(self.imu_noise) == len(self.f_n) , "Length of all logs must be the same."
        if hasattr(self, 'imu_s'):

            lengths_message = (
            f"Lengths - xhat_k_1: {len(self.xhat_k_1)}, \n"
            f"x_k: {len(self.x_k)}, \n"
            f"P_k_1: {len(self.P_k_1)}, \n"
            f"f_B: {len(self.f_B)}, \n"
            f"f_W: {len(self.f_W)}, \n"
            f"imu_s: {len(self.imu_s)}, \n"
            f"imu_noise: {len(self.imu_noise)}, \n"
            f"f_n: {len(self.f_n)}"
            )
            print(lengths_message)
            assert len(self.xhat_k_1) == len(self.x_k) == len(self.P_k_1) == len(self.f_B) == len(self.f_W) == len(self.imu_s) == len(self.imu_noise) == len(self.f_n) , "Length of all logs must be the same."
            return self.xhat_k_1 , self.x_k , self.P_k_1 , self.f_B , self.f_W , self.imu_s , self.imu_noise , self.f_n
        else:
            assert len(self.xhat_k_1) == len(self.x_k) == len(self.P_k_1) == len(self.f_B) == len(self.f_W) == len(self.imu) == len(self.imu_noise) == len(self.f_n) , "Length of all logs must be the same."
            return self.xhat_k_1 , self.x_k , self.P_k_1 , self.f_B , self.f_W , self.imu , self.imu_noise , self.f_n