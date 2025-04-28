import pytorch3d.transforms
import torch
import pytorch3d
import torch.nn.functional as F
from utils import matrix_to_quaternion


def six_dof_kin(cur_state_aug , omega_m , a_m , dT , g=-9.81 , e3=torch.tensor([0 , 0 , 1]).unsqueeze(0).t()):

    Rhat = pytorch3d.transforms.quaternion_to_matrix(cur_state_aug[0:4]).type(torch.float64)
    e3 = e3.to(Rhat.device)
    qhat = cur_state_aug[0:4]
    phat = cur_state_aug[4:7]
    vhat = cur_state_aug[7:10]
    b_omega = cur_state_aug[10:13]
    b_a = cur_state_aug[13:16]
    b_omega_w = cur_state_aug[16:19]
    b_a_w = cur_state_aug[19:22]

    a = a_m - b_a - b_a_w
    a = a.unsqueeze(0).t() # 3x1
    omega = omega_m - b_omega - b_omega_w
    omega = omega.unsqueeze(0).t() # 3x1

    row0 = torch.cat([torch.zeros(3 , 3 , device=omega.device) ,torch.eye(3, device=omega.device), torch.zeros(3 , 1, device=omega.device)] , dim=1)
    row1 = torch.cat([torch.zeros(3 , 3, device=omega.device) , torch.zeros(3 , 3, device=omega.device) , g*e3 + Rhat @ a] , dim=1)
    row2 = torch.zeros(1 , 7, device=omega.device)
    M = torch.cat([row0 , row1 , row2] , dim=0) # 7x7
    X_ = torch.cat([phat , vhat , torch.tensor([1], device=omega.device)] , dim=0) # 7
    X_ = X_.unsqueeze(0).t() # 7x1
    X_ = torch.linalg.matrix_exp(M*dT) @ X_.type(torch.float64)
    X_ = X_.squeeze() # 7

    row0 = torch.cat([torch.tensor([[0]], device=omega.device) , -omega.t()], dim=1)
    row1 = torch.cat([omega , -skew(omega.squeeze())], dim=1)
    M = .5*torch.cat([row0 , row1] , dim=0) # 4x4
    qhat = qhat.unsqueeze(0).t() # 4x1
    qhat = torch.linalg.matrix_exp(M*dT) @ qhat
    qhat = qhat.squeeze() # 4

    phat = X_[0:3]
    vhat = X_[3:6]
    qhat = pytorch3d.transforms.standardize_quaternion(qhat)
    qhat = F.normalize(qhat , p=2, dim=0)
    qhat = pytorch3d.transforms.quaternion_to_matrix(qhat)
    qhat = matrix_to_quaternion(qhat)
    qhat = F.normalize(qhat , p=2, dim=0)
    qhat = pytorch3d.transforms.standardize_quaternion(qhat)
    new_state = torch.cat([qhat , phat , vhat , b_omega , b_a] , dim=0)

    return new_state

def Vis_meas_model(cur_state , feat_pt_W):
    q = cur_state[0:4]
    p = cur_state[4:7].unsqueeze(0).t()
    assert feat_pt_W.shape[0] == 3 , "feat_pt_W must have shape (3,N)"
    R = pytorch3d.transforms.quaternion_to_matrix(q)
    z = R.t().type(torch.float64)@(feat_pt_W - p) # 3xn
    z = torch.flatten(z.T)
    return z


def skew(v):
    """
    Returns the skew-symmetric matrix of a vector v.
    :param v: A tensor of shape (3,)
    :return: A skew-symmetric matrix of shape (3, 3)
    """
    result = torch.zeros((3, 3), device=v.device)
    result[0, 1] = -v[2]
    result[0, 2] = v[1]
    result[1, 0] = v[2]
    result[1, 2] = -v[0]
    result[2, 0] = -v[1]
    result[2, 1] = v[0]
    return result
    # return torch.tensor([
    #     [0, -v[2], v[1]],
    #     [v[2], 0, -v[0]],
    #     [-v[1], v[0], 0]
    # ] , device=v.device)
    # print('for debug - delete this')
    # return torch.tensor([
    #     [0.0000, -0.0307,  0.0790],
    #     [0.0307,  0.0000,  0.7727],
    #     [-0.0790, -0.7727,  0.0000]
    # ], device=v.device)