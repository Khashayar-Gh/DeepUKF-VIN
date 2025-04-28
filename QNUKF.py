import torch
from six_dof_VIN import *
from utils import *

class QNUKF():
    def __init__(self, x0 , P0, f = six_dof_kin, h = Vis_meas_model, n = 16 , 
                 n_a = 22, alpha = 1e-4, beta = 2, kappa = 0 , IMU_dt = 1/200):
        self.device = torch.device(x0.device)
        self.f = f
        self.h = h
        self.n = n
        self.n_a = n_a
        self.dim_a = n_a - 1
        self.dim_x = n - 1
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = 3 - self.dim_a
        self.IMU_dt = IMU_dt
        self.x = x0
        self.P = P0
        self.calculate_weights()
        

        self.xa = torch.zeros(self.n_a, device=self.x.device)
        self.sigma_pts = torch.zeros(2 * self.dim_a + 1, self.n_a , device=self.x.device)
        self.sigma_pts_f = torch.zeros(2 * self.dim_a + 1, self.n, device=self.x.device)

    def update(self, feat_pts_W , feat_pts_B , h_std):
        assert feat_pts_W.shape[0] == 3 , "feat_pts_W must have shape (3,N)"
        assert feat_pts_B.shape[0] == 3 , "feat_pts_B must have shape (3,N)"
        z = torch.flatten(feat_pts_B.T)
        self.dim_z = z.shape[0]
        # self.dim_z = feat_pts_W.shape[1]*3
        self.sigma_pts_h = torch.zeros(2 * self.dim_a + 1 , self.dim_z , device=z.device)
        for i in range(2 * self.dim_a + 1):
            self.sigma_pts_h[i] = self.h(self.sigma_pts_f[i] , feat_pts_W)
            # print(self.sigma_pts_h[i] - z)

        self.calculate_z_sigma_points_stats(h_std)
        self.compute_cross_covariance()

        # self.K = self.Pxz @ torch.linalg.inv(self.Pz)
        try:
            self.K = self.Pxz @ self.Pz.pinverse()
        except:
            self.K = self.Pxz @ self.Pz
            print("Pz is singular and DELETE THIS MESSAGE AFTER DEBUGGING")
        
        self.x =  xPdelx(self.x, (self.K.type(torch.float64) @ (z - self.zhat).unsqueeze(1)).squeeze(1))
        self.P = self.P - self.K @ self.Pz @ self.K.t()
        self.P = symmetrizeCovariance(self.P)

    def predict(self, omega_m , a_m , omega_std , a_std , omega_bias_std , a_bias_std):
        self.xa = torch.cat([self.x , torch.zeros(self.n_a - self.n , device=self.x.device)])
        self.Pa = torch.block_diag(self.P  , torch.diag(omega_std**2) , torch.diag(a_std**2))
        self.calculate_sigma_points()
        self.sigma_pts_f = torch.zeros(2 * self.dim_a + 1, self.n , device=self.x.device)
        for i in range(2 * self.dim_a + 1):
            self.sigma_pts_f[i] = self.f(self.sigma_pts[i] , omega_m , a_m , self.IMU_dt)

        self.calculate_sigma_points_stats(omega_bias_std , a_bias_std)
    
    def batch_predict(self, omega_m_s, a_m_s, omega_std , a_std , omega_bias_std , a_bias_std):

        for i in range(omega_m_s.shape[0]):
            self.predict(omega_m_s[i], a_m_s[i], omega_std , a_std , omega_bias_std , a_bias_std)
        

    def calculate_weights(self):
        self.Wm = torch.zeros(2 * self.dim_a + 1 , device=self.x.device)
        self.Wc = torch.zeros(2 * self.dim_a + 1, device=self.x.device)
        self.Wm[0] = self.lam / (self.dim_a + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wm[1:] = 1 / (2 * (self.dim_a + self.lam))
        self.Wc[1:] = 1 / (2 * (self.dim_a + self.lam))

    def calculate_sigma_points(self):
        U = pseudo_sqrt((self.dim_a + self.lam) * self.Pa)
        self.sigma_pts = torch.zeros(2 * self.dim_a + 1, self.n_a , device=self.x.device)
        self.sigma_pts[0] = self.xa
        for i in range(self.dim_a):
            self.sigma_pts[i + 1] = xPdelx(self.xa , U[: , i])
            # self.sigma_pts[self.dim_a + i + 1] = xMdelx(self.xa , U[: , i])
            self.sigma_pts[self.dim_a + i + 1] = xPdelx(self.xa , -U[: , i])

    def calculate_sigma_points_stats(self, omega_bias_std , a_bias_std):
        self.x = torch.zeros(self.n , device=self.x.device)
        self.P = torch.zeros(self.dim_x , self.dim_x, device=self.x.device)
        self.x[:4] = QWA(self.sigma_pts_f[: , :4] , self.Wm)
        for i in range(2 * self.dim_a + 1):
            self.x[4:] += self.Wm[i] * self.sigma_pts_f[i , 4:]
        for i in range(2 * self.dim_a + 1):
            # self.P[3: , 3:] += self.Wc[i] * torch.outer(self.sigma_pts_f[i , 4:] - self.x[4:] ,
            #                                              self.sigma_pts_f[i , 4:] - self.x[4:])
            q_e_ = qMq(self.sigma_pts_f[i , :4] , self.x[:4]) #3
            x_e_nonq = self.sigma_pts_f[i , 4:] - self.x[4:] #13
            x_e_ = torch.cat([q_e_ , x_e_nonq] , 0) #16
            # self.P[:3 , :3] += self.Wc[i] * torch.outer(q_e_,
            #                                              q_e_)
            self.P += self.Wc[i] * torch.outer(x_e_ , x_e_)
        self.Q = torch.zeros_like(self.P , device=self.x.device)
        self.Q[-6:  , -6:] = torch.block_diag(torch.diag(omega_bias_std**2) , torch.diag(a_bias_std**2))
        self.P += self.Q
        self.P = symmetrizeCovariance(self.P)
    
    def calculate_z_sigma_points_stats(self, h_std):
        self.R = torch.eye(self.dim_z, device=self.x.device)*h_std**2
        self.zhat = torch.zeros(self.dim_z, device=self.x.device)
        for i in range(2 * self.dim_a + 1):
            self.zhat += self.Wm[i] * self.sigma_pts_h[i]
        self.Pz = torch.zeros(self.dim_z , self.dim_z, device=self.x.device)
        for i in range(2 * self.dim_a + 1):
            z_e_ = self.sigma_pts_h[i] - self.zhat
            self.Pz += self.Wc[i] * torch.outer(z_e_ , z_e_)
        self.Pz += self.R
        self.Pz = symmetrizeCovariance(self.Pz)

    def compute_cross_covariance(self):
        self.Pxz = torch.zeros(self.dim_x , self.dim_z, device=self.x.device)
        for i in range(2 * self.dim_a + 1):
            x_e = torch.zeros(self.dim_x , device=self.device)
            x_e[:3] = qMq(self.sigma_pts_f[i , :4] , self.x[:4])
            x_e[3:] = self.sigma_pts_f[i , 4:] - self.x[4:]
            z_e_ = self.sigma_pts_h[i] - self.zhat
            self.Pxz += self.Wc[i] * torch.outer(x_e , z_e_)


def pseudo_sqrt(A, reg=1e-10):
    try:
        # Attempt SVD
        U, S, Vh = torch.linalg.svd(A)
        sqrt_S = torch.sqrt(S)
        return U @ torch.diag(sqrt_S) @ Vh
    except:
        try:
            # Add small identity matrix to regularize
            A_reg = A + reg * torch.eye(A.size(0), device=A.device)
            eigvals, eigvecs = torch.linalg.eigh(A_reg)
            sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))
            return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
            print("Error in pseudo_sqrt and DELETE THIS LINE")
        except:
            # Fallback to diagonal approximation
            diag_elements = torch.diag(A)
            sqrt_diag = torch.sqrt(torch.clamp(diag_elements, min=0))
            return torch.diag(sqrt_diag)
            print('----------------------------------------------------------------')
            print("Error in pseudo_sqrt and DELETE THIS LINE")


def xPdelx(x ,delx):
    result = torch.zeros(x.shape , device=x.device)
    result[:4] = qPr(x[:4] , delx[:3])
    result[4:] = x[4:] + delx[3:]
    return result

def xMdelx(x ,delx):
    result = torch.zeros(x.shape , device=x.device)
    result[:4] = qMr(x[:4] , delx[:3])
    result[4:] = x[4:] - delx[3:]
    return result


    

    


