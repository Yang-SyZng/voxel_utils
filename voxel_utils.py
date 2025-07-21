from typing import List, Optional, Dict
import numpy as np
import open3d as o3d
import numpy as np
from argparse import Namespace
from tqdm import tqdm

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
        self.d: float = 0.0

        self.is_plane: bool = False
        self.is_init: bool = False
  
class OctoTree:
    def __init__(self, pcd: o3d.geometry.PointCloud,
                 max_layer: int, 
                 layer: int, 
                 planer_threshold: float,
                 outliers_threshold: int):
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
        self.octo_state_: int = 0   # 0: end of tree, 1: not end
        self.leaves_: List[Optional[OctoTree]] = [None for _ in range(8)]
        
        # 离群点阈值
        self.outliers_threshold: int = outliers_threshold
        
        self.init_octo_: bool = False
    
    def init_plane(self, pcd: o3d.geometry.PointCloud):
        self.plane_ptr_.center = np.zeros((3, ), dtype=np.float64)
        self.plane_ptr_.covariance = np.zeros((3, 3), dtype=np.float64)
        self.plane_ptr_.rotation = np.zeros((3, 3), dtype=np.float64)
        self.plane_ptr_.normal = np.zeros((3, ), dtype=np.float64)
        self.plane_ptr_.radius = 0
        
        # 提取平面
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model
        
        plane_cloud = pcd.select_by_index(inliers)
        points = np.asarray(plane_cloud.points)
        N = points.shape[0]
        
        obb = plane_cloud.get_oriented_bounding_box()
        self.plane_ptr_.center = np.asarray(obb.center)
        
        # PCA主成分分析
        centered_points = points - self.plane_ptr_.center
        self.plane_ptr_.covariance = centered_points.T @ centered_points / N
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(self.plane_ptr_.covariance)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.plane_ptr_.rotation = obb.R
        
        rest_cloud = pcd.select_by_index(inliers, invert=True)
        rest_points = np.asarray(rest_cloud.points)
        self.plane_ptr_.normal = eigenvectors[:, 0]
        self.plane_ptr_.radius = np.sqrt(eigenvalues[1])
        self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)

        if len(rest_cloud.points) >= self.outliers_threshold:
            numerator = np.abs(a * rest_points[:, 0] + b * rest_points[:, 1] + c * rest_points[:, 2] + d)
            denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)
            distances = numerator / denominator
            self.plane_ptr_.is_plane = distances.mean() <= 0.02
        else:
            self.plane_ptr_.is_plane = True
        if not self.plane_ptr_.is_init:
            self.plane_ptr_.is_init = True
                
    # def init_plane(self, pcd: o3d.geometry.PointCloud):
    #     # (N, 3, 1)
    #     points = np.asarray(pcd.points)
    #     N = points.shape[0]
        
    #     self.plane_ptr_.plane_cov = np.zeros((6, 6), dtype=np.float64)
    #     self.plane_ptr_.covariance = np.zeros((3, 3), dtype=np.float64)
    #     self.plane_ptr_.rotation = np.zeros((3, 3), dtype=np.float64)
    #     self.plane_ptr_.center = np.zeros(3, dtype=np.float64)
    #     self.plane_ptr_.normal = np.zeros(3, dtype=np.float64)
    #     self.plane_ptr_.points_size = N
    #     self.plane_ptr_.radius = 0
        
    #     # (3)
    #     self.plane_ptr_.center = points.mean(axis=0)
    #     # (N, 3)
    #     centered_points = points - self.plane_ptr_.center
    #     # (3, N) @ (N, 3) -> (3, 3)
    #     self.plane_ptr_.covariance = centered_points.T @ centered_points / N
        
    #     # 特征值分解
    #     eigenvalues, eigenvectors = np.linalg.eigh(self.plane_ptr_.covariance)
    #     evals = eigenvalues  # shape: (3,)
    #     evecs = -eigenvectors  # shape: (3, 3)，每列是一个特征向量
    #     self.plane_ptr_.rotation = evecs
    #     # 找最小/中间/最大特征值的索引
    #     evals_min = np.argmin(evals)
    #     evals_max = np.argmax(evals)
    #     evals_mid = 3 - evals_min - evals_max  # 总是能拿到第三个
    #     # 提取对应的特征向量
    #     evec_min = evecs[:, evals_min]  # shape: (3,)
    #     evec_mid = evecs[:, evals_mid]
    #     evec_max = evecs[:, evals_max]
    #     if evals[evals_min] < self.planer_threshold_:
    #         self.plane_ptr_.normal = evec_min
    #         self.plane_ptr_.y_normal = evec_mid
    #         self.plane_ptr_.x_normal = evec_max
    #         self.plane_ptr_.radius = np.sqrt(evals[evals_max])
    #         self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)
    #         self.plane_ptr_.is_plane = True
    #         if not self.plane_ptr_.is_init:
    #             self.plane_ptr_.is_init = True
    #     else:
    #         if not self.plane_ptr_.is_init:
    #             self.plane_ptr_.is_init = True
    #         self.plane_ptr_.is_plane = False
    #         self.plane_ptr_.normal = -1 * evec_min
    #         self.plane_ptr_.y_normal = -evec_mid
    #         self.plane_ptr_.x_normal = -evec_max
    #         self.plane_ptr_.radius = np.sqrt(evals[evals_max])
    #         self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)
            
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
                                           planer_threshold=self.planer_threshold_,
                                           outliers_threshold=self.outliers_threshold)
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

def buildVoxelMap(args: Namespace,
                  pcd: o3d.geometry.PointCloud,
                  feat_map: Dict[VOXEL_LOC, OctoTree]
                  ) -> Dict[VOXEL_LOC, OctoTree]: 
    max_layer = args.max_layer
    voxel_size = args.voxel_size
    outliers_threshold = args.outliers_threshold
    planer_threshold = args.plannar_threshold
    
    loc_xyz = np.floor(np.asarray(pcd.points) / voxel_size).astype(np.int64)
    xyz_unique, inverse_indices, _ = np.unique(loc_xyz, return_inverse=True, return_counts=True, axis=0)
    print("voxel map size:", len(xyz_unique))
    for i, xyz in enumerate(tqdm(xyz_unique, desc="Building OctoTrees")):
        position = VOXEL_LOC(xyz)
        mask = (inverse_indices == i)
        idx = np.where(mask)[0]
        pcd_in_voxel = pcd.select_by_index(idx)
        octo_tree = OctoTree(pcd=pcd_in_voxel,
                             max_layer=max_layer,
                             layer=0,
                             planer_threshold=planer_threshold,
                             outliers_threshold=outliers_threshold)
        feat_map[position] = octo_tree
        feat_map[position].quater_length_ = voxel_size / 2
        feat_map[position].voxel_center_ = (0.5 + np.array([position.x, position.y, position.z])) * voxel_size
    
    for _, value in tqdm(feat_map.items(), desc="Initializing OctoTrees"):
        value.init_octo_tree()
        
    return feat_map