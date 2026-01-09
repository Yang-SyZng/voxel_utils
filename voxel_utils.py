from typing import List, Optional, Dict
import numpy as np
import open3d as o3d
import numpy as np
from tqdm import tqdm
from arguments import ModelParams
from utils.graphics_utils import BasicPointCloud
import torch

class VOXEL_LOC:
    def __init__(self, xyz: np.ndarray):
        self.x = int(xyz[0])
        self.y = int(xyz[1])
        self.z = int(xyz[2])

    def __eq__(self, other):
        if not isinstance(other, VOXEL_LOC):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"VOXEL_LOC({self.x}, {self.y}, {self.z})"
    
class Planes:
    def __init__(self):
        self.center = []
        self.normal = []
        self.d = []
        self.complex = []

class Plane:
    def __init__(self):
        self.center = np.zeros((3, ), dtype=np.float64)
        self.normal = np.zeros((3, ), dtype=np.float64)
        self.y_normal = np.zeros((3, ), dtype=np.float64)
        self.x_normal = np.zeros((3, ), dtype=np.float64)
        self.covariance = np.zeros((3, 3), dtype=np.float64)
        self.rotation = np.zeros((3, 3), dtype=np.float64)
        
        self.scale0: float = 0.0
        self.scale1: float = 0.0
        self.d: float = 0.0

        self.is_plane: bool = False
  
class OctoTree:
    def __init__(self, pcd: BasicPointCloud,
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
    
    def init_plane(self):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(self.pcd.points)
        p.colors = o3d.utility.Vector3dVector(self.pcd.colors)
        p.normals = o3d.utility.Vector3dVector(self.pcd.normals)

        # 提取平面
        plane_model, inliers = p.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model

        plane_cloud = self.pcd.select_by_index(inliers)
        # 1. 计算几何中心 (Center)
        points = np.asarray(plane_cloud.points)
        center = np.mean(points, axis=0)
        self.plane_ptr_.center = center

        # 1. 获取均值和协方差矩阵
        mean, cov = p.compute_mean_and_covariance()
        self.plane_ptr_.center = np.asarray(mean)

        # 2. 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 3. 核心修正：重新排序轴的顺序
        # 假设你希望：X轴=最长方向, Y轴=次长方向, Z轴=法线方向(最短)
        # eigh 默认是升序(最小的在先)，所以我们反转它
        idx = np.argsort(eigenvalues)[::-1] 
        eigenvalues = eigenvalues[idx]
        R = eigenvectors[:, idx] # 此时 R 的列分别为 [主轴, 次轴, 法线]

        # 4. 核心修正：确保 R 是右手坐标系 (Right-handed System)
        # 旋转矩阵的行列式必须为 1。如果为 -1，说明是镜像/左手系。
        if np.linalg.det(R) < 0:
            # 翻转法线轴（第三列）的方向，使其符合右手定则
            R[:, 2] = -R[:, 2]

        self.plane_ptr_.rotation = R

        rest_cloud = self.pcd.select_by_index(inliers, invert=True)
        rest_points = np.asarray(rest_cloud.points)
        self.plane_ptr_.scale0 = np.sqrt(eigenvalues[0])
        self.plane_ptr_.scale1 = np.sqrt(eigenvalues[1])
        self.plane_ptr_.normal = R[:, 2] # 现在的 Z 轴就是法线
        self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal, self.plane_ptr_.center)
        
        if len(rest_cloud.points) >= self.outliers_threshold:
            numerator = np.abs(a * rest_points[:, 0] + b * rest_points[:, 1] + c * rest_points[:, 2] + d)
            denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)
            distances = numerator / denominator  
            self.plane_ptr_.is_plane = bool(distances.mean() <= 0.03)
            if distances.mean() <= 0.03:
                self.complex = self.compute_complexity(np.asarray(self.pcd.normals))
        else:
            self.plane_ptr_.is_plane = True
            self.complex = self.compute_complexity(np.asarray(self.pcd.normals))
                    
    def init_octo_tree(self):
        if self.points_num_ > self.max_plane_update_threshold_:

            self.init_plane()

            if not self.plane_ptr_.is_plane:
                self.octo_state_ = 1
                self.cut_octo_tree()

            # 代表该体素的点云大于阈值，已经初始化过八叉树了
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

        idx_sorted = np.argsort(inverse_indices)
        
        sorted_inverse_indices = inverse_indices[idx_sorted]
        diff = np.diff(sorted_inverse_indices)
        change_points = np.where(diff > 0)[0] + 1
        
        idx_groups = np.split(idx_sorted, change_points)
        # -----------------------
        for i, xyz in enumerate(leaf_xyz_unique):
            idx = idx_groups[i]
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
                    self.leaves_[i].init_plane()
                    if self.leaves_[i].plane_ptr_.is_plane:
                        self.leaves_[i].octo_state_ = 0
                    else:
                        self.leaves_[i].octo_state_ = 1
                        self.leaves_[i].cut_octo_tree()
                    self.leaves_[i].init_octo_ = True

    def compute_complexity(self, normals: np.ndarray) -> float:
        if normals.shape[0] <= 1:
            return 0.0

        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
        mean_vector = np.mean(normals, axis=0)
        return np.linalg.norm(mean_vector)
    
class VoxelMap:
    def __init__(self, cfg: ModelParams, pcd: BasicPointCloud):
        self.max_layer = cfg.max_layer
        self.voxel_size = cfg.voxel_size
        self.outliers_threshold = cfg.outliers_threshold
        self.planer_threshold = cfg.planar_threshold
        self.feat_map: Dict[VOXEL_LOC, OctoTree] = {}
        self.feat_map_first = self.buildVoxelMap(pcd)
    
    def buildVoxelMap(self, pcd: BasicPointCloud): 
        loc_xyz = np.floor(np.asarray(pcd.points) / self.voxel_size).astype(np.int64)
        xyz_unique, inverse_indices, _ = np.unique(loc_xyz, return_inverse=True, return_counts=True, axis=0)
        print("voxel map size:", len(xyz_unique))
        idx_sorted = np.argsort(inverse_indices)
        
        sorted_inverse_indices = inverse_indices[idx_sorted]
        diff = np.diff(sorted_inverse_indices)
        change_points = np.where(diff > 0)[0] + 1
        
        idx_groups = np.split(idx_sorted, change_points)

        for i, xyz in enumerate(tqdm(xyz_unique, desc="Building OctoTrees")):
            idx = idx_groups[i]
            position = VOXEL_LOC(xyz)
            # pcd_in_voxel = BasicPointCloud(points=pcd.points[idx],
            #                                colors=pcd.colors[idx],
            #                                normals=pcd.normals[idx])
            pcd_in_voxel = o3d.geometry.PointCloud()
            pcd_in_voxel.points = o3d.utility.Vector3dVector(pcd.points[idx])
            pcd_in_voxel.colors = o3d.utility.Vector3dVector(pcd.colors[idx])
            pcd_in_voxel.normals = o3d.utility.Vector3dVector(pcd.normals[idx])
            self.feat_map[position] = OctoTree(pcd=pcd_in_voxel,
                                        max_layer=self.max_layer,
                                        layer=0,
                                        planer_threshold=self.planer_threshold,
                                        outliers_threshold=self.outliers_threshold)
            self.feat_map[position].quater_length_ = self.voxel_size / 2
            self.feat_map[position].voxel_center_ = (0.5 + np.array([position.x, position.y, position.z])) * self.voxel_size

        for _, value in tqdm(self.feat_map.items(), desc="Initializing OctoTrees"):
            value.init_octo_tree()
        
        return self.bfs_valid_nodes()

    def bfs_valid_nodes(self):
        valid_nodes = []
        for _, octo_tree in tqdm(self.feat_map.items(), desc="Building first voxel"):
            queue = [octo_tree]
            while queue:
                node = queue.pop(0)
                if node.octo_state_ == 0:
                    if node.plane_ptr_.is_plane:
                        valid_nodes.append(node)
                else:
                    for leaf in node.leaves_:
                        if leaf is not None:
                            queue.append(leaf)
        return valid_nodes

    def __len__(self):
        return len(self.feat_map)

    def __getitem__(self, key) -> Optional[OctoTree]:
        if isinstance(key, tuple) and len(key) == 3:
            x, y, z = key
            loc = VOXEL_LOC(np.array([x, y, z]))

        elif isinstance(key, VOXEL_LOC):
            loc = key
        else:
            raise TypeError("Index must be a tuple of (x, y, z) or a VOXEL_LOC object")
            
        return self.feat_map.get(loc)
    def __iter__(self):
        return iter(self.feat_map.items())
    
if __name__ == '__main__':
    pass
