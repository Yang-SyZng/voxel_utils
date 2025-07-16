from typing import List, Optional, Dict
import numpy as np
import open3d as o3d
import torch
import numpy as np
from prosci import main
class VOXEL_LOC:
    def __init__(self, xyz: np.ndarray):
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
    def __eq__(self, other):
        if not isinstance(other, VOXEL_LOC):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
class Plane:
    def __init__(self):
        self.center: np.ndarray
        self.normal: np.ndarray
        self.y_normal: np.ndarray
        self.x_normal: np.ndarray
        self.covariance: np.ndarray
        self.rotation: np.ndarray

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
        
class OctoTree:
    def __init__(self, pcd: o3d.geometry.PointCloud,
                 max_layer: int, 
                 layer: int, 
                 planer_threshold: float):
        # 体素信息
        self.voxel_center_: np.ndarray
        self.quater_length_: float = 0.0
        self.max_layer_: int = max_layer
        self.layer_: int = layer
        self.max_plane_update_threshold_: int = 5
        # 体素点云
        self.pcd = pcd
        self.points_num_ = len(pcd.points)
        # 体素平面
        self.plane_ptr_: Plane = Plane()
        self.planer_threshold_: float = planer_threshold
        
        #叶子体素
        self.octo_state_: int = 0  # 0: end of tree, 1: not end
        self.leaves_: List[Optional[OctoTree]] = [None for _ in range(8)]
        
        self.update_size_threshold_: int = 5  # 固定数值
        self.all_points_num_: int = 0
        self.new_points_num_: int = 0

        self.init_octo_: bool = False
        self.update_enable_: bool = True
        self.update_cov_enable_: bool = True
    
    def init_plane(self, pcd: o3d.geometry.PointCloud):
        # (N, 3, 1)
        points = np.asarray(pcd.points)
        N = points.shape[0]
        a, b, c, d = main(data=points)
        normal = np.array([a, b, c], dtype=np.float64)
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm != 0 else normal
        print(normal)
        # 旋转矩阵
        t = np.array([1, 0, 0])  # 选择一个非平行向量
        u = np.cross(normal, t)
        u = u / np.linalg.norm(u)  # 归一化
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)  # 归一化
        rotation = np.column_stack((u, v, normal))
        # self.plane_ptr_.covariance = np.zeros((3, 3), dtype=np.float64)
        self.plane_ptr_.rotation = rotation
        self.plane_ptr_.center = points.mean(axis=0)
        self.plane_ptr_.normal = normal
        self.plane_ptr_.points_size = N
        # self.plane_ptr_.radius = 0
        
        # # (3)
        # self.plane_ptr_.center = points.mean(axis=0)
        # # (N, 3)
        # centered_points = points - self.plane_ptr_.center
        # # (3, N) @ (N, 3) -> (3, 3)
        # self.plane_ptr_.covariance = centered_points.T @ centered_points / N
        

        # # 特征值分解
        # eigenvalues, eigenvectors = np.linalg.eigh(self.plane_ptr_.covariance)
        # evals = eigenvalues  # shape: (3,)
        # evecs = -eigenvectors  # shape: (3, 3)，每列是一个特征向量
        # self.plane_ptr_.rotation = evecs
        # # 找最小/中间/最大特征值的索引
        # evals_min = np.argmin(evals)
        # evals_max = np.argmax(evals)
        # evals_mid = 3 - evals_min - evals_max  # 总是能拿到第三个
        # # 提取对应的特征向量
        # evec_min = evecs[:, evals_min]  # shape: (3,)
        # evec_mid = evecs[:, evals_mid]
        # evec_max = evecs[:, evals_max]
        
        # if evals[evals_min] < self.planer_threshold_:
        #     self.plane_ptr_.normal = evec_min
        #     self.plane_ptr_.y_normal = evec_mid
        #     self.plane_ptr_.x_normal = evec_max
        #     self.plane_ptr_.min_eigen_value = evals[evals_min].item()
        #     self.plane_ptr_.mid_eigen_value = evals[evals_mid].item()
        #     self.plane_ptr_.max_eigen_value = evals[evals_max].item()
        #     self.plane_ptr_.radius = np.sqrt(evals[evals_max])
        #     self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)
        #     self.plane_ptr_.is_plane = True
        #     if not self.plane_ptr_.is_init:
        #         self.plane_ptr_.is_init = True

            
        # else:
        #     if not self.plane_ptr_.is_init:
        #         self.plane_ptr_.is_init = True
        #     self.plane_ptr_.is_plane = False
        #     self.plane_ptr_.normal = -1 * evec_min
        #     self.plane_ptr_.y_normal = -evec_mid
        #     self.plane_ptr_.x_normal = -evec_max
        #     self.plane_ptr_.min_eigen_value = evals[evals_min].item()
        #     self.plane_ptr_.mid_eigen_value = evals[evals_mid].item()
        #     self.plane_ptr_.max_eigen_value = evals[evals_max].item()
        #     # self.plane_ptr_.radius = np.sqrt(evals[evals_max])
        #     distances = np.linalg.norm(points - self.plane_ptr_.center, axis=1)  # 欧氏距离
        #     self.plane_ptr_.radius = distances.max()
        #     self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)
            
    def init_octo_tree(self):
        if self.points_num_ > self.max_plane_update_threshold_:
            self.init_plane(self.pcd)
            if self.plane_ptr_.is_plane:
                self.octo_state_ = 0
            else:
                self.octo_state_ = 1
                self.cut_octo_tree()
            
            self.init_octo_ = True
        
    def cut_octo_tree(self):
        if self.layer_ >= self.max_layer_:
            self.octo_state_ = 0
            return
        pcd = self.pcd
        xyz = (np.asarray(pcd.points) > self.voxel_center_).astype(int)
        # 计算子节点索引
        leafnums = 4 * xyz[:, 0] + 2 * xyz[:, 1] + xyz[:, 2]  # (N,)
        leaf_xyz_unique, inverse_indices, _ = np.unique(leafnums, return_inverse=True, return_counts=True, axis=0)
        for i, xyz in enumerate(leaf_xyz_unique):
            mask = (inverse_indices == i)
            idx = np.where(mask)[0]
            pcd_in_leaf_voxel = pcd.select_by_index(idx)

            if self.leaves_[i] is None:
                self.leaves_[i] = OctoTree(pcd=pcd_in_leaf_voxel,
                                           max_layer=self.max_layer_,
                                           layer=self.layer_ + 1,
                                           planer_threshold=self.planer_threshold_)
                xyz_leaf = np.array([
                    (i // 4) % 2,
                    (i // 2) % 2,
                    i % 2
                ], dtype=np.int32)
                self.leaves_[i].voxel_center_ = self.voxel_center_ + (2 * xyz_leaf - 1) * (self.quater_length_ / 2)
                self.leaves_[i].quater_length_ = self.quater_length_ / 2
        
        for i in range(8):
            if self.leaves_[i] is not None:
                if self.leaves_[i].points_num_ > self.leaves_[i].max_plane_update_threshold_:
                    self.leaves_[i].init_plane(self.leaves_[i].pcd)
                    if self.leaves_[i].plane_ptr_.is_plane:
                        self.leaves_[i].octo_state_ = 0
                    else:
                        self.leaves_[i].octo_state_ = 1
                        self.leaves_[i].cut_octo_tree()
                    self.leaves_[i].init_octo_ = True
                    self.leaves_[i].new_points_num_ = 0

def buildVoxelMap(pcd: o3d.geometry.PointCloud,
                  voxel_size: float, max_layer: int,
                  planer_threshold: float, 
                  feat_map: Dict[VOXEL_LOC, OctoTree]
                  ) -> Dict[VOXEL_LOC, OctoTree]: 
    
    loc_xyz = np.floor(np.asarray(pcd.points) / voxel_size).astype(np.int64)
    xyz_unique, inverse_indices, _ = np.unique(loc_xyz, return_inverse=True, return_counts=True, axis=0)
    print("voxel map size:", len(xyz_unique))
    for i, xyz in enumerate(xyz_unique):
        position = VOXEL_LOC(xyz)
        mask = (inverse_indices == i)
        idx = np.where(mask)[0]
        pcd_in_voxel = pcd.select_by_index(idx)
        octo_tree = OctoTree(pcd=pcd_in_voxel,
                             max_layer=max_layer,
                             layer=0,
                             planer_threshold=planer_threshold)
        feat_map[position] = octo_tree
        feat_map[position].quater_length_ = voxel_size / 2
        feat_map[position].voxel_center_ = (0.5 + np.array([position.x, position.y, position.z])) * voxel_size
    
    for _, value in feat_map.items():
        value.init_octo_tree()
        
    return feat_map