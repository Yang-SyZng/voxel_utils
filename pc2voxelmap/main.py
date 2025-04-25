import argparse
import sys
import torch
import math
import numpy as np
# 3D point with covariance
class pointWithCov:
    def __init__(self, point: torch.Tensor, cov: torch.Tensor):
        """
        Args:
            point (torch.Tensor): Shape (3, 1), point coordinates.
            cov (torch.Tensor): Shape (3, 3), covariance matrix.
        """
        self.point = point
        self.cov = cov

class PointXYZI:
    def __init__(self, x, y, z, intensity):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity

import torch
import numpy as np

# 假设常量
DIM_STATE = 18  # 状态维度：3(旋转) + 3(位置) + 3(速度) + 3(陀螺仪偏差) + 3(加速度计偏差) + 3(重力)
INIT_COV = 0.0000001  # 初始协方差值（根据实际代码调整）

class StatesGroup:
    def __init__(self, device="cuda"):
        """
        Initialize the state group.

        Args:
            device (str): Device to place tensors on ("cuda" or "cpu").
        """
        self.device = device
        self.dtype = torch.float

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
    device = points.device
    dtype = points.dtype
    
    # 计算距离
    rang = torch.norm(points, dim=1, keepdim=True)  # 形状 (N, 1)
    
    # 测距误差方差
    range_var = range_inc * range_inc  # 标量
    
    # 角度误差方差
    direction_var = torch.zeros(2, 2, dtype=dtype, device=device)
    angle_var = (torch.sin(torch.deg2rad(torch.tensor(degree_inc))))**2
    direction_var[0, 0] = angle_var
    direction_var[1, 1] = angle_var  # 形状 (2, 2)
    
    # 归一化方向向量
    direction = points / (rang + 1e-8)  # 形状 (N, 3)
    
    # 反对称矩阵 (direction_hat)
    direction_hat = torch.zeros(N, 3, 3, dtype=dtype, device=device)
    direction_hat[:, 0, 1] = -direction[:, 2]
    direction_hat[:, 0, 2] = direction[:, 1]
    direction_hat[:, 1, 0] = direction[:, 2]
    direction_hat[:, 1, 2] = -direction[:, 0]
    direction_hat[:, 2, 0] = -direction[:, 1]
    direction_hat[:, 2, 1] = direction[:, 0]  # 形状 (N, 3, 3)
    
    # 基向量
    base_vector1 = torch.ones(N, 3, dtype=dtype, device=device)  # 形状 (N, 3)
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

def transform_lidar(state: StatesGroup, input_cloud, device="cuda"):
    """
    Transform points from LiDAR frame to world frame.

    Args:
        state (StatesGroup): State object with rot_end (3, 3) and pos_end (3, 1).
        input_cloud: List of points.
        device (str): Device to place tensors on.

    Returns:
        list: List of PointXYZI objects in world frame.
    """
    # 提取点云的 x, y, z 和 intensity
    points_lidar = torch.tensor(
        [[p.x, p.y, p.z] for p in input_cloud], dtype=torch.float, device=device
    )  # 形状 (N, 3)
    intensities = torch.tensor(
        [p.intensity for p in input_cloud], dtype=torch.float, device=device
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

def c2v(flg_EKF_inited: bool, init_map: bool, state: StatesGroup, feats_undistort,
              ranging_cov: float, angle_cov: float, Lidar_offset_to_IMU, device="cuda"):
    
    # 将点云转换到世界坐标系
    world_lidar = transform_lidar(state, feats_undistort, device)  # 列表形式
    # 提取 feats_undistort 的点 (LiDAR 坐标系)
    points_this = torch.tensor(
        [[p.x, p.y, p.z] for p in feats_undistort], dtype=torch.float, device=device
    )  # 形状 (N, 3)

    # 提取 world_lidar 的点 (世界坐标系)
    points_world = torch.tensor(
        [[p.x, p.y, p.z] for p in world_lidar], dtype=torch.float, device=device
    )  # 形状 (N, 3)

    # 如果 z=0，设置为 0.001
    points_this[:, 2] = torch.where(points_this[:, 2] == 0, 0.001, points_this[:, 2])
    
    # 计算协方差
    covs = calcBodyCov(points_this, ranging_cov, angle_cov)  # 形状 (N, 3, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for point cloud processing and mapping')
    
    # Common parameters
    parser.add_argument('--lid_topic', type=str, default='/pointcloud',
                        help='LiDAR topic name')
    parser.add_argument('--imu_topic', type=str, default='/livox/imu',
                        help='IMU topic name')
    # Pointcloud parameters
    parser.add_argument('--file_path', type=str, 
                        default='/home/y/projections/catkin_ws_cloud2VoxelMap/points3D.ply',
                        help='Path to point cloud file')
    parser.add_argument('--file_format', type=str, default='ply',
                        choices=['pcd', 'ply'],
                        help='Point cloud file format')
    # Preprocess parameters
    parser.add_argument('--lidar_type', type=int, default=2,
                        choices=[1, 2, 3],
                        help='LiDAR type: 1 for Livox, 2 for Velodyne, 3 for L515')
    parser.add_argument('--scan_line', type=int, default=32,
                        help='Number of scan lines')
    parser.add_argument('--blind', type=int, default=0,
                        help='Blind parameter')
    parser.add_argument('--point_filter_num', type=int, default=1,
                        help='Point filter number')
    parser.add_argument('--calib_laser', type=bool, default=False,
                        help='Enable laser calibration for KITTI dataset')
    # Mapping parameters
    parser.add_argument('--down_sample_size', type=float, default=0.5,
                        help='Down sample size')
    parser.add_argument('--max_iteration', type=int, default=3,
                        help='Maximum iterations')
    parser.add_argument('--voxel_size', type=float, default=2.0,
                        help='Voxel size')
    parser.add_argument('--max_layer', type=int, default=2,
                        help='Maximum layer number')
    parser.add_argument('--layer_point_size', type=int, nargs=5, 
                        default=[5, 5, 5, 5, 5],
                        help='Layer point sizes')
    parser.add_argument('--plannar_threshold', type=float, default=0.01,
                        help='Planar threshold')
    parser.add_argument('--max_points_size', type=int, default=1000,
                        help='Maximum points size')
    parser.add_argument('--max_cov_points_size', type=int, default=1000,
                        help='Maximum covariance points size')
    # Noise model parameters
    parser.add_argument('--ranging_cov', type=float, default=0.04,
                        help='Ranging covariance')
    parser.add_argument('--angle_cov', type=float, default=0.1,
                        help='Angle covariance')
    parser.add_argument('--acc_cov_scale', type=float, default=1.0,
                        help='Accelerometer covariance scale')
    parser.add_argument('--gyr_cov_scale', type=float, default=0.5,
                        help='Gyroscope covariance scale')
    # IMU parameters
    parser.add_argument('--imu_en', type=bool, default=False,
                        help='Enable IMU')
    parser.add_argument('--extrinsic_T', type=float, nargs=3, 
                        default=[0.04165, 0.02326, -0.0284],
                        help='Extrinsic translation parameters')
    parser.add_argument('--extrinsic_R', type=float, nargs=9, 
                        default=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                        help='Extrinsic rotation matrix')
    # Visualization parameters
    parser.add_argument('--pub_voxel_map', type=bool, default=True,
                        help='Publish voxel map')
    parser.add_argument('--publish_max_voxel_layer', type=int, default=3,
                        help='Maximum voxel layer to publish')
    parser.add_argument('--pub_point_cloud', type=bool, default=False,
                        help='Publish point cloud')
    parser.add_argument('--dense_map_enable', type=bool, default=False,
                        help='Enable dense map')
    parser.add_argument('--pub_point_cloud_skip', type=int, default=1,
                        help='Point cloud publishing skip rate')
    # Result parameters
    parser.add_argument('--write_kitti_log', type=bool, default=False,
                        help='Write KITTI log')
    parser.add_argument('--result_path', type=str, 
                        default='/home/b/kitt_log.txt',
                        help='Path for result log')
    # # 检查是否有命令行参数
    # if len(sys.argv) == 1:  # 如果没有提供参数
    #     parser.print_help()  # 打印帮助信息
    args = parser.parse_args()
    # c2v(args)
    
    # 测试用例
    feats_undistort = [PointXYZI(-9.10841, -6.09084, -1.2345, 0), PointXYZI(-9.10841, -6.09084, -1.2345, 0)]
    state = StatesGroup()
    range_inc = 0.01
    degree_inc = 0.1
    transform_lidar(state, feats_undistort)