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

        self.radius: float = 0.0
        self.d: float = 0.0

        self.is_plane: bool = False
  
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
    
    def init_plane(self):
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        normals = np.asarray(self.pcd.normals)
        N = points.shape[0]

        if N < self.outliers_threshold:
            self.plane_ptr_.is_plane = False
            return
        
        self.planes_list = [] # 存储该体素内发现的所有平面
        remaining_mask = np.ones(N, dtype=bool)

        # 1. 预计算每个点的局部一致性（利用你代码中的逻辑）
        # 这里简化为：检查法向量是否属于平坦区
        
        while np.sum(remaining_mask) > self.outliers_threshold:
            # 2. 找到当前剩余点中法向量最集中的方向（强核点）
            # 统计法向量直方图或使用简单的 RANSAC 采样
            idx = np.where(remaining_mask)[0]
            seed_idx = idx[0] 
            seed_normal = normals[seed_idx]

            # 初始聚类：只选法线非常接近的点 (例如 < 5度)
            core_mask = np.dot(normals[idx], seed_normal) > np.cos(np.radians(5))
            core_indices = idx[core_mask]

            if len(core_indices) < self.outliers_threshold:
                remaining_mask[idx[0]] = False # 剔除孤立点
                continue

            # 3. 拟合理想平面
            core_pts = points[core_indices]
            center = np.mean(core_pts, axis=0)
            cov = np.cov(core_pts.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            best_normal = eigenvectors[:, 0]
            best_d = -np.dot(best_normal, center)

            # 4. 几何吸收：在所有剩余点中，寻找距离该平面极近的点
            # 不管这些点的法向量是否在渐变，只要距离够近就吸收
            all_remaining_idx = np.where(remaining_mask)[0]
            dist_to_plane = np.abs(np.dot(points[all_remaining_idx], best_normal) + best_d)
            
            # 吸收阈值：dist < 0.01m 且 法线夹角在合理范围内 (例如 < 45度，防止吸收垂直面)
            absorption_mask = (dist_to_plane < 0.01) & \
                            (np.dot(normals[all_remaining_idx], best_normal) > np.cos(np.radians(45)))
            
            final_cluster_idx = all_remaining_idx[absorption_mask]

            # 5. 保存平面
            if len(final_cluster_idx) >= self.outliers_threshold:
                p = Plane()
                p.center = np.mean(points[final_cluster_idx], axis=0)
                p.normal = best_normal
                p.d = best_d
                p.is_plane = True
                self.planes_list.append(p)
                
                # 标记这些点已处理
                remaining_mask[final_cluster_idx] = False
            else:
                remaining_mask[core_indices] = False

        self.plane_ptr_.is_plane = len(self.planes_list) > 0
        # # 如果处理完后剩下的点依然很多，说明还有复杂几何没提出来，需要继续切割八叉树
        self.is_complex_region = len(remaining_mask) > self.outliers_threshold

        # # 提取平面
        # plane_model, inliers = p.segment_plane(distance_threshold=0.01,
        #                                         ransac_n=3,
        #                                         num_iterations=1000)
        # [a, b, c, d] = plane_model

        # plane_cloud = self.pcd.select_by_index(inliers)
        # points = np.asarray(plane_cloud.points)
        # N = points.shape[0]
        # try:
        #     obb = plane_cloud.get_oriented_bounding_box()
        # except RuntimeError:
        #     # OBB 失败：说明点云退化了
        #     # 在点云上加微小扰动 (1e-6)，避免完全共面/共线
        #     jitter = np.random.normal(scale=1e-6, size=points.shape)
        #     pts_perturbed = points + jitter

        #     pc_perturbed = o3d.geometry.PointCloud()
        #     pc_perturbed.points = o3d.utility.Vector3dVector(pts_perturbed)

        #     obb = pc_perturbed.get_oriented_bounding_box()

        # self.plane_ptr_.center = np.asarray(obb.get_center())
        
        # # PCA主成分分析
        # centered_points = points - self.plane_ptr_.center
        # self.plane_ptr_.covariance = centered_points.T @ centered_points / N
        
        # # 特征值分解
        # eigenvalues, eigenvectors = np.linalg.eigh(self.plane_ptr_.covariance)
        # idx = eigenvalues.argsort()
        # eigenvalues = eigenvalues[idx]
        # eigenvectors = eigenvectors[:, idx]
        # self.plane_ptr_.rotation = obb.R
        
        # rest_cloud = self.pcd.select_by_index(inliers, invert=True)
        # rest_points = np.asarray(rest_cloud.points)
        # self.plane_ptr_.normal = eigenvectors[:, 0]
        # self.plane_ptr_.radius = np.sqrt(eigenvalues[1])
        # self.plane_ptr_.d = -np.dot(self.plane_ptr_.normal.squeeze(), self.plane_ptr_.center)
        
        # if len(rest_cloud.points) >= self.outliers_threshold:
        #     numerator = np.abs(a * rest_points[:, 0] + b * rest_points[:, 1] + c * rest_points[:, 2] + d)
        #     denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        #     distances = numerator / denominator  
        #     self.plane_ptr_.is_plane = bool(distances.mean() <= 0.03)
        #     if distances.mean() <= 0.03:
        #         self.complex = self.compute_complexity(np.asarray(self.pcd.normals))
        # else:
        #     self.plane_ptr_.is_plane = True
        #     self.complex = self.compute_complexity(np.asarray(self.pcd.normals))
                    
    def init_octo_tree(self):
        if self.points_num_ > self.max_plane_update_threshold_:

            self.init_plane()

            if not self.plane_ptr_.is_plane and self.is_complex_region:
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
