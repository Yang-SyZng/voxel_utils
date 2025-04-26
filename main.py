import argparse
import sys
import argparse
import sys
import torch
import math
import numpy as np
from typing import List, Optional, Dict
from lib.common_lib import StatesGroup, PointXYZI
from utils.voxel_map_util import pointWithCov, VOXEL_LOC, OctoTree
from utils import DOUBLE
import utils.voxel_map_util as vx
def c2v(state: StatesGroup, feats_undistort,
        ranging_cov: float, angle_cov: float, Lidar_offset_to_IMU, 
        max_layer, max_voxel_size, max_points_size, max_cov_points_size, min_eigen_value, voxel_map, 
        device="cuda"):
    
    #
    # if (flg_EKF_inited && !init_map) start
    #
    # 将点云转换到世界坐标系
    world_lidar = vx.transform_lidar(state, feats_undistort, device)  # 列表形式
    
    #
    # "for" change to "Batch" start
    #
    # 提取 feats_undistort 的点 (LiDAR 坐标系)
    points_this = torch.tensor(
        [[p.x, p.y, p.z] for p in feats_undistort], dtype=DOUBLE, device=device
    )  # 形状 (N, 3)

    # 提取 world_lidar 的点 (世界坐标系)
    points_world = torch.tensor(
        [[p.x, p.y, p.z] for p in world_lidar], dtype=DOUBLE, device=device
    )  # 形状 (N, 3)

    # 如果 z=0，设置为 0.001
    points_this[:, 2] = torch.where(points_this[:, 2] == 0, 0.001, points_this[:, 2])
    
    # 计算协方差
    covs = vx.calcBodyCov(points_this, ranging_cov, angle_cov)  # 形状 (N, 3, 3)
    # 协方差传播到世界坐标系
    points_this = points_this + Lidar_offset_to_IMU  # 形状 (N, 3)
    
    # 计算叉积矩阵
    point_crossmat = torch.zeros(points_this.shape[0], 3, 3, dtype=DOUBLE, device=device)
    point_crossmat[:, 0, 1] = -points_this[:, 2]
    point_crossmat[:, 0, 2] = points_this[:, 1]
    point_crossmat[:, 1, 0] = points_this[:, 2]
    point_crossmat[:, 1, 2] = -points_this[:, 0]
    point_crossmat[:, 2, 0] = -points_this[:, 1]
    point_crossmat[:, 2, 1] = points_this[:, 0]  # 形状 (N, 3, 3)
    
    
    # 提取状态协方差的子块
    rot_end = state.rot_end  # 形状 (3, 3)
    cov_rot = state.cov[:3, :3]  # 形状 (3, 3)
    cov_pos = state.cov[3:6, 3:6]  # 形状 (3, 3)
    # 协方差传播
    N = points_this.shape[0]
    # (N, 3, 3) * (N, 3, 3) * (N, 3, 3)
    term1 = torch.bmm(torch.bmm(rot_end.unsqueeze(0).expand(N, -1, -1), covs),
                        rot_end.t().unsqueeze(0).expand(N, -1, -1))  # (N, 3, 3)
    term2 = torch.bmm(torch.bmm(-point_crossmat, cov_rot.unsqueeze(0).expand(N, -1, -1)),
                        (-point_crossmat).transpose(1, 2))  # (N, 3, 3)
    term3 = cov_pos.unsqueeze(0).expand(N, -1, -1)  # (N, 3, 3)
    covs = term1 + term2 + term3  # 形状 (N, 3, 3)
    # 创建 pv_list
    pv_list = []
    for i in range(points_world.shape[0]):
        pv = pointWithCov(
            point=points_world[i].reshape(3),  # 形状 (3)
            cov=covs[i]  # 形状 (3, 3)
        )
        pv_list.append(pv)

    # 计算标准差
    sigma_pv = torch.diagonal(covs, dim1=1, dim2=2)  # 形状 (N, 3)
    sigma_pv = torch.sqrt(sigma_pv)  # 形状 (N, 3)
    #
    # "for" change to "Batch" end
    #
    print("pv_list size:", len(pv_list))
    # print("max_layer:", max_layer)

    vx.buildVoxelMap(pv_list, max_voxel_size, max_layer,
                max_points_size, max_points_size, min_eigen_value, voxel_map)
    #
    # if (flg_EKF_inited && !init_map) end
    #
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
    feats_undistort = [PointXYZI(-9.10841, -6.09084, -1.2345, 0.0), PointXYZI(-9.10841, -6.09084, -1.2345, 0.0)]
    state = StatesGroup()
    range_inc = 0.04
    degree_inc = 0.1
    Lidar_offset_to_IMU = torch.tensor([[0, 0, 0]], dtype=DOUBLE, device="cuda")
    max_layer = 2
    max_voxel_size = 2
    max_points_size = 1000
    max_cov_points_size = 1000
    min_eigen_value = 0.01
    voxel_map: Dict[VOXEL_LOC, OctoTree] = {}
    c2v(state, feats_undistort, range_inc, degree_inc, Lidar_offset_to_IMU,
        max_layer, max_voxel_size, max_points_size, max_cov_points_size, min_eigen_value, voxel_map)