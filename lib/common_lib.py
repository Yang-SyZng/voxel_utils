import torch
from utils import DOUBLE, DEVICE
from lib import DIM_STATE, INIT_COV

#   CCCCCCCCC\    LL\            AAAAAAAA\      SSSSSSSS\     SSSSSSSS\
#  CC ________|   LL |          AA  ____AA\    SS  ______|   SS  ______|
#  CC |           LL |          AA /    AA |   SS /          SS /
#  CC |           LL |          AAAAAAAAAA |     SSSSSSS \     SSSSSSS \
#  CC |           LL |          AA  ____AA |           SS \           SS \
#  CC |           LL |          AA |    AA |           SS |           SS |
#   CCCCCCCCC\    LLLLLLLLLL\   AA |    AA |    SSSSSSSS /     SSSSSSSS /
#   \_________|   \_________|   \__|    \__|    \_______/      \_______/
# Created by zty 2025/04/26

class PointXYZI:
    def __init__(self, x, y, z, intensity):
        self.x = torch.tensor(x, dtype=DOUBLE, device=DEVICE)
        self.y = torch.tensor(y, dtype=DOUBLE, device=DEVICE)
        self.z = torch.tensor(z, dtype=DOUBLE, device=DEVICE)
        self.intensity = torch.tensor(intensity, dtype=torch.int, device=DEVICE)
        
class PointXYZINormal:
    def __init__(self, x, y, z, intensity, normal):
        self.x = torch.tensor(x, dtype=DOUBLE, device=DEVICE)
        self.y = torch.tensor(y, dtype=DOUBLE, device=DEVICE)
        self.z = torch.tensor(z, dtype=DOUBLE, device=DEVICE)
        self.intensity = torch.tensor(intensity, dtype=torch.int, device=DEVICE)
        self.normal = torch.tensor(normal, dtype=torch.int, device=DEVICE)

class StatesGroup:
    def __init__(self, device="cuda"):
        """
        Initialize the state group.

        Args:
            device (str): Device to place tensors on ("cuda" or "cpu").
        """
        self.device = device
        self.dtype = DOUBLE

        # 初始化状态
        self.rot_end = torch.eye(3, dtype=self.dtype, device=device)  # 形状 (3, 3)
        self.pos_end = torch.zeros(3, 1, dtype=self.dtype, device=device)  # 形状 (3, 1)
        self.vel_end = torch.zeros(3, 1, dtype=self.dtype, device=device)  # 形状 (3, 1)
        self.bias_g = torch.zeros(3, 1, dtype=self.dtype, device=device)  # 形状 (3, 1)
        self.bias_a = torch.zeros(3, 1, dtype=self.dtype, device=device)  # 形状 (3, 1)
        self.gravity = torch.zeros(3, 1, dtype=self.dtype, device=device)  # 形状 (3, 1)
        self.cov = torch.eye(DIM_STATE, dtype=self.dtype, device=device) * INIT_COV  # 形状 (18, 18)
    def __add__(self, state_add):
        """
        Add a state increment to the current state.

        Args:
            state_add (torch.Tensor): Shape (DIM_STATE, 1), state increment.

        Returns:
            StatesGroup: New state group with updated values.
        """
        new_state = StatesGroup(device=self.device)

        # 旋转增量：Exp mapping（李代数到李群）
        theta = state_add[:3].flatten()  # 形状 (3,)
        theta_norm = torch.norm(theta)
        if theta_norm < 1e-8:
            rot_increment = torch.eye(3, dtype=self.dtype, device=self.device)
        else:
            # 罗德里格斯公式
            skew_theta = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
            skew_theta[0, 1] = -theta[2]
            skew_theta[0, 2] = theta[1]
            skew_theta[1, 0] = theta[2]
            skew_theta[1, 2] = -theta[0]
            skew_theta[2, 0] = -theta[1]
            skew_theta[2, 1] = theta[0]
            rot_increment = (torch.eye(3, dtype=self.dtype, device=self.device) +
                             torch.sin(theta_norm) / theta_norm * skew_theta +
                             (1 - torch.cos(theta_norm)) / (theta_norm ** 2) * skew_theta @ skew_theta)
        new_state.rot_end = self.rot_end @ rot_increment

        # 其他状态直接加法
        new_state.pos_end = self.pos_end + state_add[3:6].reshape(3, 1)
        new_state.vel_end = self.vel_end + state_add[6:9].reshape(3, 1)
        new_state.bias_g = self.bias_g + state_add[9:12].reshape(3, 1)
        new_state.bias_a = self.bias_a + state_add[12:15].reshape(3, 1)
        new_state.gravity = self.gravity + state_add[15:18].reshape(3, 1)
        new_state.cov = self.cov.clone()

        return new_state

    def __iadd__(self, state_add):
        """
        In-place addition of a state increment.

        Args:
            state_add (torch.Tensor): Shape (DIM_STATE, 1), state increment.

        Returns:
            StatesGroup: Self with updated values.
        """
        # 旋转增量
        theta = state_add[:3].flatten()
        theta_norm = torch.norm(theta)
        if theta_norm < 1e-8:
            rot_increment = torch.eye(3, dtype=self.dtype, device=self.device)
        else:
            skew_theta = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
            skew_theta[0, 1] = -theta[2]
            skew_theta[0, 2] = theta[1]
            skew_theta[1, 0] = theta[2]
            skew_theta[1, 2] = -theta[0]
            skew_theta[2, 0] = -theta[1]
            skew_theta[2, 1] = theta[0]
            rot_increment = (torch.eye(3, dtype=self.dtype, device=self.device) +
                             torch.sin(theta_norm) / theta_norm * skew_theta +
                             (1 - torch.cos(theta_norm)) / (theta_norm ** 2) * skew_theta @ skew_theta)
        self.rot_end = self.rot_end @ rot_increment

        # 其他状态
        self.pos_end += state_add[3:6].reshape(3, 1)
        self.vel_end += state_add[6:9].reshape(3, 1)
        self.bias_g += state_add[9:12].reshape(3, 1)
        self.bias_a += state_add[12:15-DIM_STATE].reshape(3, 1)
        self.gravity += state_add[15:18].reshape(3, 1)

        return self

    def __sub__(self, other):
        """
        Compute the difference between two states.

        Args:
            other (StatesGroup): Another state group.

        Returns:
            torch.Tensor: Shape (DIM_STATE, 1), state difference.
        """
        diff = torch.zeros(DIM_STATE, 1, dtype=self.dtype, device=self.device)

        # 旋转差：Log mapping（李群到李代数）
        rotd = (other.rot_end.t() @ self.rot_end)
        trace = torch.trace(rotd)
        if trace > 3 - 1e-8:
            theta = torch.zeros(3, dtype=self.dtype, device=self.device)
        else:
            theta_norm = torch.acos((trace - 1) / 2)
            log_rot = theta_norm / (2 * torch.sin(theta_norm)) * (rotd - rotd.t())
            theta = torch.tensor([
                log_rot[2, 1],  # theta_x
                log_rot[0, 2],  # theta_y
                log_rot[1, 0]   # theta_z
            ], dtype=self.dtype, device=self.device)
        diff[:3, 0] = theta

        # 其他状态差
        diff[3:6] = self.pos_end - other.pos_end
        diff[6:9] = self.vel_end - other.vel_end
        diff[9:12] = self.bias_g - other.bias_g
        diff[12:15] = self.bias_a - other.bias_a
        diff[15:18] = self.gravity - other.gravity

        return diff

    def resetpose(self):
        """
        Reset pose-related states to zero.
        """
        self.rot_end = torch.eye(3, dtype=self.dtype, device=self.device)
        self.pos_end = torch.zeros(3, 1, dtype=self.dtype, device=self.device)
        self.vel_end = torch.zeros(3, 1, dtype=self.dtype, device=self.device)


# FFFFFFFF\    UU\     UU\    NN\    NN\     CCCCCCCCC\    TTTTTTTTTT\   IIIIII\      OOOOOOOO\      NN\     NN\     SSSSSSSS\
# FF  _____|   UU |    UU |   NNN\   NN |   CC ________|       TT  __|     II  _|    OO _____OO \    NNN\    NN |   SS  ______|
# FF |         UU |    UU |   NN NN  NN |   CC |               TT |        II |     OO /      OO |   NNNN\   NN |   SS /
# FFFFF\       UU |    UU |   NN \N\ NN |   CC |               TT |        II |     OO |      OO |   NN NN\  NN |    SSSSSSS \
# FF  __|      UU |    UU |   NN |\NNNN |   CC |               TT |        II |     OO |      OO |   NN | NN\NN |           SS \
# FF |         UU |    UU |   NN | \NNN |   CC |               TT |        II |      OO \    OO /    NN |  NNNN |           SS |
# FF |         \UUUUUUUU /    NN |  \NN |    CCCCCCCCC\        TT |      IIIIII\      OOOOOOOO /     NN |   NNN |    SSSSSSSS /
# \__|          \_______/     \__|   \__|    \_________|       \__|      \______|     \_______|      \__|   \___|    \_______/
# Created by zty 2025/04/26

import torch

def Exp(v1: float, v2: float, v3: float) -> torch.Tensor:
    """
    计算 3x3 旋转矩阵，使用Roderigous Tranformation。

    Args:
        v1, v2, v3 (float): 3D 向量分量。

    Returns:
        torch.Tensor: 3x3 旋转矩阵。
    """
    # 将输入转换为张量
    v = torch.tensor([v1, v2, v3], dtype=DOUBLE, device=DEVICE).reshape(3, 1)
    
    # 计算范数
    norm = torch.norm(v)
    
    # 初始化单位矩阵
    Eye3 = torch.eye(3, dtype=DOUBLE, device=DEVICE)
    
    if norm > 0.00001:
        # 归一化向量
        r_ang = v / norm
        
        # 构造反对称矩阵 K
        K = torch.tensor([
            [0, -r_ang[2, 0], r_ang[1, 0]],
            [r_ang[2, 0], 0, -r_ang[0, 0]],
            [-r_ang[1, 0], r_ang[0, 0], 0]
        ], dtype=DOUBLE, device=DEVICE)
        
        # 罗德里格斯公式
        return Eye3 + torch.sin(norm) * K + (1.0 - torch.cos(norm)) * K @ K
    else:
        return Eye3
    
def only_propag(meas: MeasureGroup, state_inout: StatesGroup, pcl_out: 'PointXYZI') -> None:
        pcl_beg_time = meas.lidar_beg_time

        # 设置输出点云
        pcl_out.points = meas.lidar.points
        pcl_out.num_points = meas.lidar.num_points

        # 计算点云结束时间
        if len(pcl_out) > 0:
            pcl_end_time = pcl_beg_time + pcl_out.points[-1, 3] / 1000.0
        else:
            pcl_end_time = pcl_beg_time

        # 计算时间差 dt
        if b_first_frame_:
            dt = 0.1
            b_first_frame_ = False
            time_last_scan_ = pcl_beg_time
        else:
            dt = pcl_beg_time - time_last_scan_
            time_last_scan_ = pcl_beg_time

        # 协方差传播
        F_x = torch.eye(DIM_STATE, dtype=DOUBLE, device=DEVICE)
        cov_w = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)

        # 计算旋转增量
        v1, v2, v3 = state_inout.bias_g[0, 0], state_inout.bias_g[1, 0], state_inout.bias_g[2, 0]
        Exp_f = Exp(v1, v2, v3, dt, device=DEVICE)  # 使用新的 Exp 函数

        # 设置 F_x 的子块
        F_x[0:3, 0:3] = Exp_f  # 旋转部分
        F_x[0:3, 9:12] = torch.eye(3, dtype=DOUBLE, device=DEVICE) * dt
        F_x[3:6, 6:9] = torch.eye(3, dtype=DOUBLE, device=DEVICE) * dt

        # 设置噪声协方差 cov_w
        cov_w[9:12, 9:12] = torch.eye(3, dtype=DOUBLE, device=DEVICE) * cov_gyr * (dt ** 2)
        cov_w[6:9, 6:9] = torch.eye(3, dtype=DOUBLE, device=DEVICE) * cov_acc * (dt ** 2)

        # 更新协方差
        state_inout.cov = F_x @ state_inout.cov @ F_x.transpose(0, 1) + cov_w

        # 更新状态
        state_inout.rot_end = state_inout.rot_end @ Exp_f
        state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt