import torch
from pytorch3d.transforms import quaternion_to_matrix
from six_dof_VIN import skew , six_dof_kin , Vis_meas_model
from torch.linalg import matrix_exp as expm
from utils import symmetrizeCovariance , qPq
from QNUKF import xPdelx


class ErrorStateEKF:
    def __init__(self, x0, P0,dt =1/200,state_dim=16,error_state_dim=15):
        self.state_dim = state_dim
        self.error_state_dim = error_state_dim
        self.dt = dt
        
        # Initialize the state and covariance matrices
        self.x = x0
        self.P = P0
        self.device = x0.device
        
        # Process and measurement noise covariance
        
    
    def predict(self, omega_m, a_m, omega_std , a_std , omega_bias_std , a_bias_std):
        """
        Prediction step of EKF.
        control_input: control vector (torch tensor)
        F: State transition matrix (torch tensor)
        B: Control matrix (torch tensor)
        """
        # Predict state
        self.Q = torch.block_diag(*[torch.diag(omega_std**2) , torch.diag(a_std**2) ,
                                     torch.diag(omega_bias_std**2) , torch.diag(a_bias_std**2)])
        self.Q[6:12, 6:12] = self.Q[6:12, 6:12] / (self.dt ** 2)
        Phi_k, Q_k = self.FG(omega_m, a_m)
        self.x = six_dof_kin_ekf(self.x , omega_m , a_m , self.dt)        
        # Predict covariance
        self.P = torch.matmul(Phi_k, torch.matmul(self.P, Phi_k.T)) + Q_k
        self.P = symmetrizeCovariance(self.P)
    
    def batch_predict(self, omega_m_s, a_m_s, omega_std , a_std , omega_bias_std , a_bias_std):

        for i in range(omega_m_s.shape[0]):
            self.predict(omega_m_s[i], a_m_s[i], omega_std , a_std , omega_bias_std , a_bias_std)

    def update(self, feat_pts_W , feat_pts_B , h_std):
        """
        Update step of EKF.
        measurement: actual measurement (torch tensor)
        H: Measurement matrix (torch tensor)
        z_pred: Predicted measurement (torch tensor)
        """
        # Innovation (residual)
        
        z_pred = Vis_meas_model(self.x , feat_pts_W)
        z = torch.flatten(feat_pts_B.T)
        self.dim_z = z.shape[0]
        self.R = torch.eye(self.dim_z, device=self.device)*h_std**2
        y = z - z_pred
        H = self.cal_H(feat_pts_W)
        # Innovation covariance
        S = torch.matmul(H, torch.matmul(self.P, H.T)) + self.R
        S = symmetrizeCovariance(S)
        # Kalman gain
        P_xz = torch.matmul(self.P, H.T)
        K = torch.matmul(P_xz, torch.inverse(S))
        # K = torch.matmul(self.P, torch.matmul(H.T, torch.inverse(S)))
        
        # Update state
        self.x = xPdelx_ekf(self.x , torch.matmul(K, y))
        
        # Update covariance
        I = torch.eye(self.error_state_dim, device=self.device)
        self.P = torch.matmul((I - torch.matmul(K, H)), self.P)
        self.P = symmetrizeCovariance(self.P)
    
    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.covariance
    
    def FG(self, omega_m, a_m):
    # cur_state = [q, p, v, bw, ba]
        cur_state = self.x
        Rhat = quaternion_to_matrix(cur_state[0:4])
        
        bw = cur_state[10:13]
        ba = cur_state[13:16]
        
        omega = omega_m - bw
        a = a_m - ba
        
        zeros3 = torch.zeros(3, 3 , device=self.device)
        eye3 = torch.eye(3, device=self.device)
        
        F_ = torch.cat([
            torch.cat([-skew(omega), zeros3, zeros3, -eye3, zeros3], dim=1),
            torch.cat([zeros3, zeros3, eye3, zeros3, zeros3], dim=1),
            torch.cat([-Rhat @ skew(a), zeros3, zeros3, zeros3, -Rhat], dim=1),
            torch.cat([zeros3, zeros3, zeros3, zeros3, zeros3], dim=1),
            torch.cat([zeros3, zeros3, zeros3, zeros3, zeros3], dim=1)
        ], dim=0)
        
        G_ = torch.cat([
            torch.cat([-eye3, zeros3, zeros3, zeros3], dim=1),
            torch.cat([zeros3, zeros3, zeros3, zeros3], dim=1),
            torch.cat([zeros3, -Rhat, zeros3, zeros3], dim=1),
            torch.cat([zeros3, zeros3, eye3, zeros3], dim=1),
            torch.cat([zeros3, zeros3, zeros3, eye3], dim=1)
        ], dim=0)
        
        dT = self.dt
        Phi_k = expm(F_ * dT)
        
        Q = self.Q
        
        def fun(tou):
            return expm(F_ * (dT - tou)) @ G_
        
        def fun2(tou):
            G_tou = fun(tou)
            return G_tou @ Q @ G_tou.T
        
        # Discretizing time steps for torch.trapezoid
        num_steps = 100
        tou_vals = torch.linspace(0, dT, num_steps, device=self.device)
        integrand_vals = torch.stack([fun2(tou) for tou in tou_vals])
        
        # Using trapezoidal rule for integration
        Q_k = torch.trapezoid(integrand_vals, tou_vals, dim=0)

        return Phi_k, Q_k
    
    def cal_H(self, P_I):
        # x = [q, p ,...]
        x = self.x
        point_count = self.dim_z//3
        Rhat = quaternion_to_matrix(x[0:4])
        phat = x[4:7]
        
        H = torch.zeros(self.dim_z, 15 , device=self.device)
        
        for i in range(point_count):
            # Update H matrix for each point
            H[3*i:3*i+3, 0:3] = -Rhat.T @ skew(P_I[:, i] - phat)
            H[3*i:3*i+3, 3:6] = -Rhat.T

        return H
    

def six_dof_kin_ekf(cur_state , omega_m , a_m , dT , g=-9.81 ,
                     e3=torch.tensor([0 , 0 , 1]).unsqueeze(0).t()):
    cur_state_aug = torch.cat([cur_state, torch.zeros(6 , device=cur_state.device)], dim=0)
    return six_dof_kin(cur_state_aug , omega_m , a_m , dT , g, e3)

def xPdelx_ekf(x ,delx):
    result = torch.zeros(x.shape , device=x.device)
    result[:4] = qPq(x[:4] , torch.cat((torch.tensor([1], device=x.device) , 0.5*delx[:3])))
    result[4:] = x[4:] + delx[3:]
    return result

