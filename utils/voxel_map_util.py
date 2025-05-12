import torch
from typing import List, Optional, Dict
from lib.common_lib import StatesGroup, PointCloudXYZINormal, PointCloudXYZI, PointCloudXYZ, BasedPointCloud
from utils import DOUBLE, DEVICE
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
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

# pointWithCov [x, y, z][covs]
class pointWithCov:
    def __init__(self, points: torch.Tensor=None, covs: torch.Tensor=None): # , curvature=None
        if points is None or covs is None:
            self.points = torch.zeros((0, 3), dtype=DOUBLE, device=DEVICE)  # (N, 3)
            self.covs = torch.zeros((0, 3, 3), dtype=DOUBLE, device=DEVICE) # (N, 3, 3)
        else:
            if not isinstance(points, torch.Tensor):
                raise TypeError("must be a torch.Tensor")
            if not isinstance(covs, torch.Tensor):
                raise TypeError("must be a torch.Tensor")
            if points.shape[1] != 3:
                raise ValueError("points must have shape (N, 3) for [x, y, z]")
            self.points = points.to(dtype=DOUBLE, device=DEVICE)
            self.covs = covs.to(dtype=DOUBLE, device=DEVICE)

    def add_points(self, points, covs=None):
        if not isinstance(points, torch.Tensor):
            raise TypeError("must be a torch.Tensor")
        if points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3) for [x, y, z]")
        self.points = torch.cat([self.points, points.to(device=DEVICE)], dim=0)
        if covs is not None:
            if not isinstance(covs, torch.Tensor):
                raise TypeError("must be a torch.Tensor")
            self.covs = torch.cat([self.covs, covs.to(device=DEVICE)], dim=0)

    @property
    def size(self):
        return self.points.shape[0]

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
        return hash((self.x, self.y, self.z))
    
class OctoTree:
    def __init__(self, 
                 max_layer: int, 
                 layer: int, 
                 layer_point_size: List[int],
                 max_points_size: int, 
                 max_cov_points_size: int, 
                 planer_threshold: float):
        self.temp_points_ = pointWithCov()
        self.new_points_ = PointCloudXYZ()
        self.plane_ptr_: Plane = Plane()
        self.max_layer_: int = max_layer
        self.layer_: int = layer
        self.octo_state_: int = 0  # 0: end of tree, 1: not end
        self.layer_point_size_: List[int] = layer_point_size
        self.leaves_: List[OctoTree] = [None for _ in range(8)]
        
        self.voxel_center_: torch.Tensor = torch.zeros((3), dtype=DOUBLE, device=DEVICE)  # 修正为张量  # x, y, z
        
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
        

    
    def init_plane(self, points: pointWithCov):
        global plane_id
        
        self.plane_ptr_.plane_cov = torch.zeros((6, 6), dtype=DOUBLE, device=DEVICE)
        self.plane_ptr_.covariance = torch.zeros((3, 3), dtype=DOUBLE, device=DEVICE)
        self.plane_ptr_.center = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
        self.plane_ptr_.normal = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
        self.plane_ptr_.points_size = points.points.shape[0]
        self.plane_ptr_.radius = 0
        
        # (N, 3, 1)
        points_tensor = points.points
        N = points.points.shape[0]
        # (3)
        self.plane_ptr_.center = torch.mean(points_tensor, dim=0)
        # (N, 3)
        centered_points = points_tensor - self.plane_ptr_.center
        # (3, N) @ (N, 3) -> (3, 3)
        self.plane_ptr_.covariance = torch.matmul(centered_points.T, centered_points) / N
        
        # print(self.plane_ptr_.center)
        # print(self.plane_ptr_.covariance)
        # exit(-1)
        # 特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(self.plane_ptr_.covariance)
        evals = eigenvalues  # 特征值对角矩阵
        evecs = -eigenvectors # 特征向量
        
        # 找到最小、中间、最大特征值的索引
        evals_min = torch.argmin(evals)
        evals_max = torch.argmax(evals)
        evals_mid = 3 - evals_min - evals_max
        evec_min = evecs[:, evals_min].reshape(3, 1)
        evec_mid = evecs[:, evals_mid].reshape(3, 1)
        evec_max = evecs[:, evals_max].reshape(3, 1)
        
        # 平面协方差计算（保持不变）
        J_Q = torch.tensor([[1.0 / N, 0.0, 0.0],
                            [0.0, 1.0 / N, 0.0],
                            [0.0, 0.0, 1.0 / N]], dtype=DOUBLE, device=DEVICE)
        
        if evals[evals_min] < self.planer_threshold_:
            F = torch.zeros((N, 3, 3), dtype=DOUBLE, device=DEVICE)  # (N, 3, 3)
            for m in range(3):
                if m != evals_min:
                    denom = (evals[evals_min] - evals[m]) * N
                    evec_m = evecs[:, m].reshape(3, 1)
                    term = evec_m @ evec_min.T + evec_min @ evec_m.T
                    # (N, 1, 3) @ (3, 3) -> (N, 1, 3)
                    F_m = (centered_points.unsqueeze(1) / denom) @ term  # (N, 1, 3)
                    F[:, m, :] = F_m.squeeze(1)  # (N, 3)
                else:
                    F[:, m, :] = torch.zeros((N, 3), dtype=DOUBLE, device=DEVICE)

            J = torch.zeros((N, 6, 3), dtype=DOUBLE, device=DEVICE)
            J[:, :3, :] = torch.matmul(evecs, F)  # (N, 3, 3)
            J[:, 3:6, :] = J_Q.unsqueeze(0).repeat(N, 1, 1)  # (N, 3, 3)
            # (N, 6, 3) @ (N, 3, 3) -> (N, 6, 3)
            temp = torch.matmul(J, points.covs)
            # (N, 6, 3) @ (N, 3, 6) -> (N, 6, 6)
            plane_cov_per_point = torch.matmul(temp, J.transpose(1, 2))  # (N, 6, 6)
            self.plane_ptr_.plane_cov = torch.sum(plane_cov_per_point, dim=0)  # (6, 6)

            self.plane_ptr_.normal = evec_min
            self.plane_ptr_.y_normal = evec_mid
            self.plane_ptr_.x_normal = evec_max
            self.plane_ptr_.min_eigen_value = evals[evals_min].item()
            self.plane_ptr_.mid_eigen_value = evals[evals_mid].item()
            self.plane_ptr_.max_eigen_value = evals[evals_max].item()
            self.plane_ptr_.radius = torch.sqrt(evals[evals_max]).item()
            self.plane_ptr_.d = -(torch.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)).item()
            self.plane_ptr_.is_plane = True
            
            if self.plane_ptr_.last_update_points_size == 0:
                self.plane_ptr_.last_update_points_size = self.plane_ptr_.points_size
                self.plane_ptr_.is_update = True
            elif self.plane_ptr_.points_size - self.plane_ptr_.last_update_points_size > 100:
                self.plane_ptr_.last_update_points_size = self.plane_ptr_.points_size
                self.plane_ptr_.is_update = True

            if not self.plane_ptr_.is_init:
                self.plane_ptr_.id = plane_id
                plane_id += 1
                self.plane_ptr_.is_init = True

        else:
            if not self.plane_ptr_.is_init:
                self.plane_ptr_.id = plane_id
                plane_id += 1
                self.plane_ptr_.is_init = True
            
            if self.plane_ptr_.last_update_points_size == 0:
                self.plane_ptr_.last_update_points_size = self.plane_ptr_.points_size
                self.plane_ptr_.is_update = True
            elif self.plane_ptr_.points_size - self.plane_ptr_.last_update_points_size > 100:
                self.plane_ptr_.last_update_points_size = self.plane_ptr_.points_size
                self.plane_ptr_.is_update = True
            
            self.plane_ptr_.is_plane = False
            self.plane_ptr_.normal = -1 * evec_min
            self.plane_ptr_.y_normal = -evec_mid
            self.plane_ptr_.x_normal = -evec_max
            self.plane_ptr_.min_eigen_value = evals[evals_min].item()
            self.plane_ptr_.mid_eigen_value = evals[evals_mid].item()
            self.plane_ptr_.max_eigen_value = evals[evals_max].item()
            self.plane_ptr_.radius = torch.sqrt(evals[evals_max]).item()
            self.plane_ptr_.d = -(torch.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)).item()
        
    def init_octo_tree(self):
        if self.temp_points_.points.shape[0] > self.max_plane_update_threshold_:
            self.init_plane(self.temp_points_)
            if self.plane_ptr_.is_plane:
                self.octo_state_ = 0
                if self.temp_points_.points.shape[0] > self.max_cov_points_size_:
                    self.update_cov_enable_ = False
                if self.temp_points_.points.shape[0] > self.max_points_size_:
                    self.update_enable_ = False
            else:
                self.octo_state_ = 1
                self.cut_octo_tree()
            
            self.init_octo_ = True
            self.new_points_num_ = 0
        
    def cut_octo_tree(self):
        if self.layer_ >= self.max_layer_:
            self.octo_state_ = 0
            return
        points_tensor = self.temp_points_.points
        xyz = (points_tensor > self.voxel_center_).int()  # (N, 3)
        # 计算子节点索引
        leafnums = 4 * xyz[:, 0] + 2 * xyz[:, 1] + xyz[:, 2]  # (N,)
        for leafnum in torch.unique(leafnums):
            leafnum_int: int = leafnum.item()
            mask = (leafnums == leafnum_int)  # 筛选属于当前子节点的点
            points_in_leaf = pointWithCov(self.temp_points_.points[mask], covs=self.temp_points_.covs[mask])
            if self.leaves_[leafnum_int] is None:
                self.leaves_[leafnum_int] = OctoTree(
                    max_layer=self.max_layer_,
                    layer=self.layer_ + 1,
                    layer_point_size=self.layer_point_size_,
                    max_points_size=self.max_points_size_,
                    max_cov_points_size=self.max_cov_points_size_,
                    planer_threshold=self.planer_threshold_
                )
                xyz_leaf = torch.tensor([
                    (leafnum_int // 4) % 2,
                    (leafnum_int // 2) % 2,
                    leafnum_int % 2
                ], dtype=torch.int32, device=DEVICE)
                self.leaves_[leafnum_int].voxel_center_ = self.voxel_center_ + (2 * xyz_leaf - 1) * self.quater_length_
                self.leaves_[leafnum_int].quater_length_ = self.quater_length_ / 2

            # 分配点到子节点
            self.leaves_[leafnum_int].temp_points_.add_points(points_in_leaf.points, points_in_leaf.covs)
            self.leaves_[leafnum_int].new_points_num_ += points_in_leaf.size
        
        for i in range(8):
            if self.leaves_[i] is not None:
                if self.leaves_[i].temp_points_.size > self.leaves_[i].max_plane_update_threshold_:
                    self.leaves_[i].init_plane(self.leaves_[i].temp_points_)
                    if self.leaves_[i].plane_ptr_.is_plane:
                        self.leaves_[i].octo_state_ = 0
                    else:
                        self.leaves_[i].octo_state_ = 1
                        self.leaves_[i].cut_octo_tree()
                    self.leaves_[i].init_octo_ = True
                    self.leaves_[i].new_points_num_ = 0
    def UpdateOctoTree(self):
        pass

class PointCloud:
    def __init__(self, points: np.ndarray = None, timestamps: np.ndarray = None):
        self.points = points  # 形状 (N, 3)，numpy 数组
        self.timestamps = timestamps  # 形状 (N,)，numpy 数组，表示时间戳
        self.size = len(points) if points is not None else 0

    def to_open3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd
    
# FFFFFFFF\    UU\     UU\    NN\    NN\     CCCCCCCCC\    TTTTTTTTTT\   IIIIII\      OOOOOOOO\      NN\     NN\     SSSSSSSS\
# FF  _____|   UU |    UU |   NNN\   NN |   CC ________|       TT  __|     II  _|    OO _____OO \    NNN\    NN |   SS  ______|
# FF |         UU |    UU |   NN NN  NN |   CC |               TT |        II |     OO /      OO |   NNNN\   NN |   SS /
# FFFFF\       UU |    UU |   NN \N\ NN |   CC |               TT |        II |     OO |      OO |   NN NN\  NN |    SSSSSSS \
# FF  __|      UU |    UU |   NN |\NNNN |   CC |               TT |        II |     OO |      OO |   NN | NN\NN |           SS \
# FF |         UU |    UU |   NN | \NNN |   CC |               TT |        II |      OO \    OO /    NN |  NNNN |           SS |
# FF |          UUUUUUUU /    NN |  \NN |    CCCCCCCCC\        TT |      IIIIII\      OOOOOOO /      NN |   NNN |    SSSSSSSS /
# \__|          \_______/     \__|   \__|    \_________|       \__|      \______|     \______|       \__|   \___|    \_______/
# Created by zty 2025/04/26
def downsample_point_cloud(input_cloud: PointCloudXYZINormal, voxel_size: float = 0.05) -> PointCloudXYZINormal:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(input_cloud.points[:, :3].clone().cpu().numpy())
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_points = np.asarray(down_pcd.points)  # (M, 3)
    down_points = torch.tensor(down_points, dtype=DOUBLE, device=DEVICE)
    intensity = torch.zeros((down_points.shape[0], 1), dtype=DOUBLE, device=DEVICE)
    curvature = torch.zeros((down_points.shape[0], 1), dtype=DOUBLE, device=DEVICE)
    normals_tensor = torch.zeros((down_points.shape[0], 3), dtype=DOUBLE, device=DEVICE)
    return PointCloudXYZINormal(
        points=torch.cat([
                down_points,  # (N, 3) -> [x, y, z]
                intensity,      # (N, 1) -> [intensity]
                normals_tensor, # (N, 3) -> [nx, ny, nz]
                curvature       # (N, 1) -> [curvature]
            ], dim=1))  # 结果形状 (N, 7)    )


def GetUpdatePlane(current_octo: OctoTree, pub_max_voxel_layer: int, plane_list: List[Plane]):
    if current_octo.layer_ > pub_max_voxel_layer:
        return plane_list
    if current_octo.plane_ptr_.is_update:
        plane_list.append(current_octo.plane_ptr_)
    if current_octo.layer_ < current_octo.max_layer_ and not current_octo.plane_ptr_.is_plane:
        for i in range(0, 8):
            if current_octo.leaves_[i] is not None:
                GetUpdatePlane(current_octo.leaves_[i], pub_max_voxel_layer, plane_list)

def pubVoxelMap(voxel_map: Dict[VOXEL_LOC, OctoTree], pub_max_voxel_layer: int):
    max_trace = 0.25
    pow_num = 0.2
    use_alpha = 0.8
    pub_plane_list: List[Plane] = []
    for _, value in voxel_map.items():
        GetUpdatePlane(value, pub_max_voxel_layer, pub_plane_list)
    print("voxel_map_plane_size:", len(pub_plane_list))
    
    # 感觉下面是在RVIZ上写的
    # for pub_plane in pub_plane_list:
    #     plane_cov_trace = torch.trace(pub_plane.plane_cov[:3, :3])
    #     if plane_cov_trace >= max_trace:
    #         plane_cov_trace = max_trace
    #     plane_cov_trace *= (1.0 / max_trace)
    #     plane_cov_trace = plane_cov_trace ** pow_num
    #     r, g, b = mapJet(plane_cov_trace, 0, 1)
    #     plane_rgb = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
    #     plane_rgb[0] = r / 256.0, plane_rgb[1] = g / 256.0, plane_rgb[2] = b / 256.0
    #     alpha = use_alpha if pub_plane.is_plane else alpha = 0
    #     pubSinglePlane(voxel_map, "plane", pub_plane, alpha, plane_rgb)
        
def pubSinglePlane():
    pass

def buildVoxelMap(input_points: pointWithCov,
                  voxel_size: float, max_layer: int,
                  layer_point_size: List[int],
                  max_points_size: int, max_cov_points_size: int, 
                  planer_threshold: float, 
                  feat_map: Dict[VOXEL_LOC, OctoTree], 
                  ) -> Dict[VOXEL_LOC, OctoTree]: 
    # input_points.points.shape (N, 3)
    loc_xyz = input_points.points / voxel_size # (N, 3)
    loc_xyz = torch.where(loc_xyz < 0, loc_xyz - 1.0, loc_xyz)  # 处理负数
    loc_xyz = loc_xyz.to(dtype=torch.int64, device=DEVICE)
    
    loc_xyz_unique, inverse_indices = torch.unique(loc_xyz, dim=0, return_inverse=True)
    N_unique = loc_xyz_unique.shape[0]
    print("voxel map size:", N_unique)
    # 为每个体素创建或更新 OctoTree
    for i in range(N_unique):
        voxel_idx = loc_xyz_unique[i]  # (3,)
        position = VOXEL_LOC(int(voxel_idx[0]), int(voxel_idx[1]), int(voxel_idx[2]))
        
        # 找到属于该体素的所有点
        mask = (inverse_indices == i)
        points_in_voxel = input_points.points[mask]
        covs_in_voxel = input_points.covs[mask]
        
        # 创建体素
        octo_tree = OctoTree(max_layer, 0, layer_point_size, max_points_size,
                            max_cov_points_size, planer_threshold)
        feat_map[position] = octo_tree
        feat_map[position].quater_length_ = voxel_size / 4
        feat_map[position].voxel_center_[0] = (0.5 + position.x) * voxel_size
        feat_map[position].voxel_center_[1] = (0.5 + position.y) * voxel_size
        feat_map[position].voxel_center_[2] = (0.5 + position.z) * voxel_size
        feat_map[position].temp_points_.add_points(points_in_voxel, covs_in_voxel)
        feat_map[position].new_points_num_ += len(points_in_voxel)
        feat_map[position].layer_point_size_ = layer_point_size

    # test
    # position = VOXEL_LOC(-5, -4, -1)
    # print(feat_map[position].quater_length_)
    # print(feat_map[position].voxel_center_)
    # print(feat_map[position].temp_points_.points[0])
    # print(feat_map[position].temp_points_.covs[0])
    # print(feat_map[position].new_points_num_)
    # exit(-1)
    
    # 初始化所有 OctoTree
    for _, value in feat_map.items():
        value.init_octo_tree()
    # position = VOXEL_LOC(-5, -4, -1)
    # print(feat_map[position].plane_ptr_.center)
    # print(feat_map[position].plane_ptr_.covariance)
    # print(feat_map[position].plane_ptr_.normal)
    # print(feat_map[position].plane_ptr_.x_normal)
    # print(feat_map[position].plane_ptr_.y_normal)
    # print(feat_map[position].plane_ptr_.plane_cov)
    # print(feat_map[position].plane_ptr_.radius)
    # print(feat_map[position].plane_ptr_.min_eigen_value)
    # print(feat_map[position].plane_ptr_.mid_eigen_value)
    # print(feat_map[position].plane_ptr_.max_eigen_value)
    # print(feat_map[position].plane_ptr_.d)
    # print(feat_map[position].plane_ptr_.points_size)
    # exit(-1)
    return feat_map

def transformLidar(state: StatesGroup, input_cloud: PointCloudXYZINormal) -> PointCloudXYZI:
    """
    Transform points from LiDAR frame to world frame.

    Args:
        state (StatesGroup): State object with rot_end (3, 3) and pos_end (3, 1).
        input_cloud: List of PointCloudXYZI objects

    Returns:
        list: List of PointCloudXYZI objects in world frame.
    """
    # 提取点云的 x, y, z 和 intensity
    points_lidar = input_cloud.points[:, :3]  # 形状 (N, 3)
    intensities = input_cloud.points[:, 3].reshape(-1, 1)  # 形状 (N, 1)

    # state.rot_end 和 state.pos_end 已经是张量
    rot_end = state.rot_end  # 形状 (3, 3)
    pos_end = state.pos_end  # 形状 (3, 1)

    # 变换：world = rot_end @ lidar + pos_end
    #               (3, 3)   (3, N) + (3, 1)
    
    points_world = (rot_end @ points_lidar.T + pos_end).T  # 形状 (N, 3)
    
    # 创建输出点云列表
    trans_cloud = PointCloudXYZI()
    points = torch.cat([
                points_world,  # (N, 3) -> [x, y, z]
                intensities,      # (N, 1) -> [intensity]
                ], dim=1)  # 结果形状 (N, 4)
    trans_cloud.add_points(points)
    
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
    
    # 计算距离
    rang = torch.norm(points, dim=1, keepdim=True)  # 形状 (N, 1)
    # 测距误差方差
    range_var = range_inc * range_inc  # 标量
    # 角度误差方差
    direction_var = torch.zeros(2, 2, dtype=DOUBLE, device=DEVICE)
    angle_var = (torch.sin(torch.deg2rad(torch.tensor(degree_inc))))**2
    direction_var[0, 0] = angle_var
    direction_var[1, 1] = angle_var  # 形状 (2, 2)
    # 归一化方向向量
    direction = points / rang   # 形状 (N, 3)
    # 反对称矩阵 (direction_hat)
    direction_hat = torch.zeros(N, 3, 3, dtype=DOUBLE, device=DEVICE)
    direction_hat[:, 0, 1] = -direction[:, 2]
    direction_hat[:, 0, 2] = direction[:, 1]
    direction_hat[:, 1, 0] = direction[:, 2]
    direction_hat[:, 1, 2] = -direction[:, 0]
    direction_hat[:, 2, 0] = -direction[:, 1]
    direction_hat[:, 2, 1] = direction[:, 0]  # 形状 (N, 3, 3)
    
    # 基向量
    base_vector1 = torch.ones(N, 3, dtype=DOUBLE, device=DEVICE)  # 形状 (N, 3)
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
def mapJet(v: float, vmin: float, vmax: float):
    v = vmin if v < vmin else None
    v = vmax if v > vmax else None
    if v < 0.1242:
        db = 0.504 + ((1. - 0.504) / 0.1242) * v
        dg = dr = 0.
    elif v < 0.3747:
        db = 1.
        dr = 0.
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242))
    elif v < 0.6253:
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747))
        dg = 1.
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747))
    elif v < 0.8758:
        db = 0.
        dr = 1.
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253))
    else:
        db = dg = 0.
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758))
    
    r = int(255 * dr)
    g = int(255 * dg)
    b = int(255 * db)
    
    return r, g, b