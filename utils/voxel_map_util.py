import torch
from typing import List, Optional, Dict
from lib.common_lib import StatesGroup, PointXYZI
from utils import DOUBLE, HASH_P, MAX_N, DEVICE

#  VV \        VV \   AAAAAAAA\    LL\          UU\     UU\   EEEEEEEEEEE\  SSSSSSSS\
#   VV \      VV /   AA  ____AA\   LL |         UU |    UU |  EE  ______|  SS  ______|
#    VV \    VV /    AA /    AA |  LL |         UU |    UU |  EE |         SS /
#     VV \  VV /     AAAAAAAAAA |  LL |         UU |    UU |  EEEEEEEEEE\    SSSSSSS \
#      VV \VV /      AA  ____AA |  LL |         UU |    UU |  EE  ______|           SS \
#       VVVV /       AA |    AA |  LL |         UU |    UU |  EE |                  SS |
#        VV /        AA |    AA |  LLLLLLLLLL\   UUUUUUUU /   EEEEEEEEEEE\   SSSSSSSS /
#        \_/         \__|    \__|  \_________|   \_______/    \__________|   \_______/
# Created by zty 2025/05/03

plane_id = 0

#   CCCCCCCCC\    LL\            AAAAAAAA\      SSSSSSSS\     SSSSSSSS\     EEEEEEEEEEE\  SSSSSSSS\
#  CC ________|   LL |          AA  ____AA\    SS  ______|   SS  ______|    EE  ______|  SS  ______|
#  CC |           LL |          AA /    AA |   SS /          SS /           EE |         SS /
#  CC |           LL |          AAAAAAAAAA |     SSSSSSS \     SSSSSSS \    EEEEEEEEEE\    SSSSSSS \
#  CC |           LL |          AA  ____AA |           SS \           SS \  EE  ______|           SS \
#  CC |           LL |          AA |    AA |           SS |           SS |  EE |                  SS |
#   CCCCCCCCC\    LLLLLLLLLL\   AA |    AA |    SSSSSSSS /     SSSSSSSS /   EEEEEEEEEEE\   SSSSSSSS /
#   \_________|   \_________|   \__|    \__|    \_______/      \_______/    \__________|   \_______/
# Created by zty 2025/04/26

class Ptpl:
    def __init__(self):
        self.point: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
        self.normal: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
        self.center: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
        self.plane_cov: torch.Tensor = torch.zeros((6, 6), dtype=DOUBLE, device=DEVICE)
        self.d: float = 0.0
        self.layer: int = 0

class pointWithCov:
    def __init__(self, point: torch.Tensor, cov: torch.Tensor):
        """
        Args:
            point (torch.Tensor): Shape (3, 1), point coordinates.
            cov (torch.Tensor): Shape (3, 3), covariance matrix.
        """
        self.point = point
        self.cov = cov

class Plane:
    def __init__(self):
        # 
        self.center: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)       
        self.normal: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)       
        self.y_normal: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)     
        self.x_normal: torch.Tensor = torch.zeros(3, dtype=DOUBLE, device=DEVICE)     
        self.covariance: torch.Tensor = torch.zeros((3, 3), dtype=DOUBLE, device=DEVICE)  
        self.plane_cov: torch.Tensor = torch.zeros((6, 6), dtype=DOUBLE, device=DEVICE)   

        self.radius: float = 0.0
        self.min_eigen_value: float = 1.0
        self.mid_eigen_value: float = 1.0
        self.max_eigen_value: float = 1.0
        self.d: float = 0.0
        self.points_size: int = 0

        self.is_plane: bool = False
        self.is_init: bool = False
        self.id: Optional[int] = None  # 没默认值的地方，用Optional

        # 只用于发布Plane
        self.is_update: bool = False
        self.last_update_points_size: int = 0
        self.update_enable: bool = True


class VOXEL_LOC:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        if not isinstance(other, VOXEL_LOC):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return ((((self.z) * HASH_P) % MAX_N + (self.y)) * HASH_P) % MAX_N + (self.x)
    
class OctoTree:
    def __init__(self, 
                 max_layer: int, 
                 layer: int, 
                 layer_point_size: List[int],
                 max_points_size: int, 
                 max_cov_points_size: int, 
                 planer_threshold: float):
        self.temp_points_: List[pointWithCov] = []
        self.new_points_: List[pointWithCov] = []
        self.plane_ptr_: Plane = Plane()
        self.max_layer_: int = max_layer
        self.layer_: int = layer
        self.octo_state_: int = 0  # 0: end of tree, 1: not end
        self.layer_point_size_: List[int] = layer_point_size
        self.leaves_: List[Optional['OctoTree']] = [None for _ in range(8)]
        
        self.voxel_center_: List[float] = [0.0, 0.0, 0.0]  # x, y, z
        
        self.quater_length_: float = 0.0
        self.planer_threshold_: float = planer_threshold
        self.max_plane_update_threshold_: int = self.layer_point_size_[self.layer_]
        self.update_size_threshold_: int = 5  # 固定数值
        self.all_points_num_: int = 0
        self.new_points_num_: int = 0
        self.max_points_size_: int = max_points_size
        self.max_cov_points_size_: int = max_cov_points_size
        self.init_octo_: bool = False
        self.update_enable_: bool = True
        self.update_cov_enable_: bool = True
        

    
    def init_plane(self, points: List[pointWithCov], plane: Plane):
        global plane_id
        
        plane.plane_cov = torch.zeros((6, 6), dtype=DOUBLE, device=DEVICE)
        plane.covariance = torch.zeros((3, 3), dtype=DOUBLE, device=DEVICE)
        plane.center = torch.zeros((3), dtype=DOUBLE, device=DEVICE)
        plane.normal = torch.zeros((3), dtype=DOUBLE, device=DEVICE)
        plane.points_size = len(points)
        plane.radius = 0
        
        # 计算重心和协方差
        for pv in points:
            plane.covariance += pv.point @ pv.point.T
            plane.center += pv.point
        plane.center = plane.center / plane.points_size
        plane.covariance = plane.covariance / plane.points_size - \
                           plane.center * plane.center.T
        
        # 特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(plane.covariance)
        evals = eigenvalues  # 特征值对角矩阵
        evecs = eigenvectors # 特征向量
        
        # 找到最小、中间、最大特征值的索引
        evals_min = torch.argmin(evals)
        evals_max = torch.argmax(evals)
        evals_mid = 3 - evals_min - evals_max
        evec_min = evecs[:, evals_min].reshape(3, 1)
        evec_mid = evecs[:, evals_mid].reshape(3, 1)
        evec_max = evecs[:, evals_max].reshape(3, 1)
        
        # 平面协方差计算（保持不变）
        J_Q = torch.tensor([[1.0 / plane.points_size, 0.0, 0.0],
                            [0.0, 1.0 / plane.points_size, 0.0],
                            [0.0, 0.0, 1.0 / plane.points_size]], dtype=DOUBLE, device=DEVICE)
        
        if evals[evals_min] < self.planer_threshold_:
            for i in range(len(points)):
                J = torch.zeros(6, 3, dtype=DOUBLE, device=DEVICE)
                F = torch.zeros(3, 3, dtype=DOUBLE, device=DEVICE)
                for m in range(3):
                    if m != evals_min:
                        F_m = ((points[i].point - plane.center).t() /
                            (plane.points_size * (evals[evals_min] - evals[m]))) @ \
                            (evecs[:, m].reshape(3, 1) @ evecs[:, evals_min].reshape(1, 3) +
                            evecs[:, evals_min].reshape(3, 1) @ evecs[:, m].reshape(1, 3))
                        F[m, :] = F_m
                    else:
                        F[m, :] = torch.zeros(3, dtype=DOUBLE, device=DEVICE)

                J[:3, :] = evecs @ F
                J[3:6, :] = J_Q
                plane.plane_cov += J @ points[i].cov @ J.t()
            plane.normal = evec_min
            plane.y_normal = evec_mid
            plane.x_normal = evec_max
            plane.min_eigen_value = evals[evals_min].item()
            plane.mid_eigen_value = evals[evals_mid].item()
            plane.max_eigen_value = evals[evals_max].item()
            plane.radius = torch.sqrt(evals[evals_max]).item()
            plane.d = -(plane.normal[0] * plane.center[0] +
                        plane.normal[1] * plane.center[1] +
                        plane.normal[2] * plane.center[2]).item()
            plane.is_plane = True
            
            if plane.last_update_points_size == 0:
                plane.last_update_points_size = plane.points_size
                plane.is_update = True
            elif plane.points_size - plane.last_update_points_size > 100:
                plane.last_update_points_size = plane.points_size
                plane.is_update = True

            if not plane.is_init:
                plane.id = plane_id
                plane_id += 1
                plane.is_init = True

        else:
            if not plane.is_init:
                plane.id = plane_id
                plane_id += 1
                plane.is_init = True
            
            if plane.last_update_points_size == 0:
                plane.last_update_points_size = plane.points_size
                plane.is_update = True
            elif plane.points_size - plane.last_update_points_size > 100:
                plane.last_update_points_size = plane.points_size
                plane.is_update = True
            
            plane.is_plane = False
            plane.normal = evec_min
            plane.y_normal = evec_mid
            plane.x_normal = evec_max
            plane.min_eigen_value = evals[evals_min].item()
            plane.mid_eigen_value = evals[evals_mid].item()
            plane.max_eigen_value = evals[evals_max].item()
            plane.radius = torch.sqrt(evals[evals_max]).item()
            plane.d = -(plane.normal[0] * plane.center[0] +
                        plane.normal[1] * plane.center[1] +
                        plane.normal[2] * plane.center[2]).item()
        
    def init_octo_tree(self):
        if len(self.temp_points_) < self.max_plane_update_threshold_:
            self.init_plane(self.temp_points_, self.plane_ptr_)
            if self.plane_ptr_.is_plane:
                self.octo_state_ = 0
                if len(self.temp_points_) > self.max_cov_points_size_:
                    self.update_cov_enable_ = False
                if len(self.temp_points_) > self.max_points_size_:
                    self.update_enable_ = False
        else:
            self.octo_state_ = 1
            self.cut_octo_tree()
        
        self.init_octo_ = True
        self.new_points_num_ = 0
        
    def cut_octo_tree(self):
        pass
    
    def UpdateOctoTree(self):
        pass


# FFFFFFFF\    UU\     UU\    NN\    NN\     CCCCCCCCC\    TTTTTTTTTT\   IIIIII\      OOOOOOOO\      NN\     NN\     SSSSSSSS\
# FF  _____|   UU |    UU |   NNN\   NN |   CC ________|       TT  __|     II  _|    OO _____OO \    NNN\    NN |   SS  ______|
# FF |         UU |    UU |   NN NN  NN |   CC |               TT |        II |     OO /      OO |   NNNN\   NN |   SS /
# FFFFF\       UU |    UU |   NN \N\ NN |   CC |               TT |        II |     OO |      OO |   NN NN\  NN |    SSSSSSS \
# FF  __|      UU |    UU |   NN |\NNNN |   CC |               TT |        II |     OO |      OO |   NN | NN\NN |           SS \
# FF |         UU |    UU |   NN | \NNN |   CC |               TT |        II |      OO \    OO /    NN |  NNNN |           SS |
# FF |          UUUUUUUU /    NN |  \NN |    CCCCCCCCC\        TT |      IIIIII\      OOOOOOO /      NN |   NNN |    SSSSSSSS /
# \__|          \_______/     \__|   \__|    \_________|       \__|      \______|     \______|       \__|   \___|    \_______/
# Created by zty 2025/04/26

def buildVoxelMap(input_points: List[pointWithCov], voxel_size: float, max_layer: int,
                  layer_point_size,
                  max_points_size: int,
                  max_cov_points_size: int, planer_threshold: float, 
                  feat_map: Dict[VOXEL_LOC, OctoTree], 
                  ):
    for p_v in input_points:
        loc_xyz = torch.zeros((3), dtype=DOUBLE, device=DEVICE)
        for i in range(3):
            loc_xyz[i] = p_v.point[i] / voxel_size
            if loc_xyz[i] < 0: 
                loc_xyz[i] = loc_xyz[i] - 1.0
        position = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
        if position in feat_map:
            feat_map[position].temp_points_.append(p_v)
            feat_map[position].new_points_ += 1
        else:
            octo_tree = OctoTree(max_layer, 0, layer_point_size, max_points_size,
                                 max_cov_points_size, planer_threshold)
            feat_map[position] = octo_tree
            feat_map[position].quater_length_ = voxel_size / 4
            feat_map[position].voxel_center_[0] = (0.5 + position.x) * voxel_size
            feat_map[position].voxel_center_[1] = (0.5 + position.y) * voxel_size
            feat_map[position].voxel_center_[2] = (0.5 + position.z) * voxel_size
            feat_map[position].temp_points_.append(p_v)
            feat_map[position].new_points_num_ += 1
            feat_map[position].layer_point_size_ = layer_point_size
        
    for _, value in feat_map.items():
        value.init_octo_tree()

def transform_lidar(state: StatesGroup, input_cloud):
    """
    Transform points from LiDAR frame to world frame.

    Args:
        state (StatesGroup): State object with rot_end (3, 3) and pos_end (3, 1).
        input_cloud: List of points.

    Returns:
        list: List of PointXYZI objects in world frame.
    """
    # 提取点云的 x, y, z 和 intensity
    points_lidar = torch.tensor(
        [[p.x, p.y, p.z] for p in input_cloud], dtype=DOUBLE, device=DEVICE
    )  # 形状 (N, 3)
    intensities = torch.tensor(
        [p.intensity for p in input_cloud], dtype=DOUBLE, device=DEVICE
    )  # 形状 (N,)

    # state.rot_end 和 state.pos_end 已经是张量
    rot_end = state.rot_end  # 形状 (3, 3)
    pos_end = state.pos_end  # 形状 (3, 1)

    # 变换：world = rot_end @ lidar + pos_end
    points_world = (rot_end @ points_lidar.t() + pos_end).t()  # 形状 (N, 3)

    # 创建输出点云列表
    trans_cloud = []
    points_world = points_world.cpu().numpy()
    intensities = intensities.cpu().numpy()
    for i in range(points_world.shape[0]):
        pi = PointXYZI(
            x=points_world[i, 0],
            y=points_world[i, 1],
            z=points_world[i, 2],
            intensity=intensities[i]
        )
        trans_cloud.append(pi)
    
    return trans_cloud


def calcBodyCov(points: torch.Tensor, range_inc: float, degree_inc: float) -> torch.Tensor:
    """
    Description: Using (N, 3, 1) Ver.
    
    Args:
        points (torch.Tensor): Shape (N, 3), N 3D points.
        range_inc (float): Range uncertainty increment.
        degree_inc (float): Angular uncertainty increment in degrees.

    Returns:
        torch.Tensor: Covariance matrices of shape (N, 3, 3).
    """
    # exacting property
    N = points.shape[0]
    dtype = points.dtype
    
    # 计算距离
    rang = torch.norm(points, dim=1, keepdim=True)  # 形状 (N, 1)
    # 测距误差方差
    range_var = range_inc * range_inc  # 标量
    # 角度误差方差
    direction_var = torch.zeros(2, 2, dtype=dtype, device=DEVICE)
    angle_var = (torch.sin(torch.deg2rad(torch.tensor(degree_inc))))**2
    direction_var[0, 0] = angle_var
    direction_var[1, 1] = angle_var  # 形状 (2, 2)
    # 归一化方向向量
    direction = points / rang   # 形状 (N, 3)
    # 反对称矩阵 (direction_hat)
    direction_hat = torch.zeros(N, 3, 3, dtype=dtype, device=DEVICE)
    direction_hat[:, 0, 1] = -direction[:, 2]
    direction_hat[:, 0, 2] = direction[:, 1]
    direction_hat[:, 1, 0] = direction[:, 2]
    direction_hat[:, 1, 2] = -direction[:, 0]
    direction_hat[:, 2, 0] = -direction[:, 1]
    direction_hat[:, 2, 1] = direction[:, 0]  # 形状 (N, 3, 3)
    
    # 基向量
    base_vector1 = torch.ones(N, 3, dtype=dtype, device=DEVICE)  # 形状 (N, 3)
    base_vector1[:, 0] = 1.0
    base_vector1[:, 1] = 1.0
    base_vector1[:, 2] = -(direction[:, 0] + direction[:, 1]) / (direction[:, 2] + 1e-8)
    base_vector1 = base_vector1 / (torch.norm(base_vector1, dim=1, keepdim=True) + 1e-8)  # 形状 (N, 3)

    base_vector2 = torch.cross(base_vector1, direction, dim=1)  # 形状 (N, 3)
    base_vector2 = base_vector2 / (torch.norm(base_vector2, dim=1, keepdim=True) + 1e-8)
    
    # 矩阵 N
    N = torch.stack([base_vector1, base_vector2], dim=2)  # 形状 (N, 3, 2)

    # 矩阵 A
    A = rang.unsqueeze(-1) * torch.bmm(direction_hat, N)  # (N, 3, 3) @ (N, 3, 2) -> (N, 3, 2)

    # 协方差矩阵
    direction = direction.unsqueeze(2)  # (N, 3, 1)
    direction_t = direction.transpose(1, 2)  # (N, 1, 3)
    
    # term1 = torch.einsum("n i k, n k l, n k j -> n i j", direction, range_var, direction_t)        
    term1 = torch.bmm(direction, range_var * direction_t)  # (N, 3, 1) @ (N, 1, 3) -> (N, 3, 3)
    term2 = torch.einsum("n i k, k l, n l j -> n i j", A, direction_var, A.transpose(1, 2)) # (N, 3, 2) @ (2, 2) @ (N, 2, 3) -> (N, 3, 3)
    cov = term1 + term2  # 形状 (N, 3, 3)

    return cov