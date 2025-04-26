import torch
from utils import DOUBLE
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
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.intensity = float(intensity)


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