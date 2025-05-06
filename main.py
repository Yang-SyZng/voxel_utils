import argparse
from argparse import Namespace
import yaml
import torch
from typing import Final, List, Dict
import lib.common_lib as cl
from lib.common_lib import StatesGroup, PointXYZI, ImuProcess # PointXYZINormal
from lib import DIM_STATE
from utils.voxel_map_util import pointWithCov, VOXEL_LOC, OctoTree
from utils import DOUBLE, DEVICE
import utils.voxel_map_util as vx
import open3d as o3d
INIT_TIME: Final = 0.0
CALIB_ANGLE_COV: Final = 0.01


feats_undistort: List[PointXYZI] = []
feats_down_body: List[PointXYZI] = []
laserCloudOri: List[PointXYZI] = []
laserCloudNoeffect: List[PointXYZI] = []

def read_yaml(yaml_path: str):
    """读取 YAML 配置文件，并转成 argparse.Namespace"""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 将多层嵌套展开成一个平面字典
    flat_cfg = {}

    def flatten(d, parent_key=''):
        for k, v in d.items():
            new_key = k if parent_key == '' else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_cfg[new_key] = v

    flatten(cfg)

    return Namespace(**flat_cfg)

def readPointCloud(file_path: str, file_format: str,):
    if file_format not in ["pcd", "ply"]:
        raise ValueError(f"Unsupported file format: {file_format}")

    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        raise ValueError(f"Couldn't read {file_format.upper()} file {file_path}: {str(e)}")

    if not pcd.has_points():
        raise ValueError(f"Loaded point cloud is empty: {file_path}")

    return pcd

def main(*args: Namespace):
    if isinstance(args, tuple):
        args = args[0]
    # cummon params
    lid_topic = args.lid_topic
    imu_topic = args.imu_topic
    
    # noise model params
    ranging_cov = args.ranging_cov
    angle_cov = args.angle_cov
    gyr_cov_scale = args.gyr_cov_scale
    acc_cov_scale = args.acc_cov_scale
    
    # imu params, current version does not support imu
    imu_en = args.imu_en
    extrinT = torch.tensor(args.extrinsic_T, dtype=DOUBLE, device=DEVICE).reshape(3, 1)
    extrinR = torch.tensor(args.extrinsic_R, dtype=DOUBLE, device=DEVICE).reshape(3, 3)
    
    # mapping algorithm params
    NUM_MAX_ITERATIONS = args.max_iteration
    max_points_size = args.max_points_size
    max_cov_points_size = args.max_cov_points_size
    layer_point_size = args.layer_point_size
    max_layer = args.max_layer
    max_voxel_size = args.voxel_size
    filter_size_surf_min = args.down_sample_size
    plannar_threshold = args.plannar_threshold # min_eigen_value
    
    # preprocess params
    # bline = args.blind # pre->bind
    calib_laser = args.calib_laser 
    # lidar_type = args.lidar_type # p_pre->lidar_type
    scan_line = args.scan_line # p_pre->N_SCANS
    
    # visualization params
    publish_voxel_map = args.pub_voxel_map
    publish_max_voxel_layer = args.publish_max_voxel_layer
    publish_point_cloud = args.pub_point_cloud
    pub_point_cloud_skip = args.pub_point_cloud_skip
    dense_map_en = args.dense_map_enable
    
    # result params
    write_kitti_log = args.write_kitti_log
    result_path = args.result_path
    
    file_path = args.file_path
    file_format = args.file_format
    
    pcd = readPointCloud(file_path, file_format)
    
    # solution: 18x1 列向量
    solution = torch.zeros((DIM_STATE, 1), dtype=DOUBLE, device=DEVICE)

    # G, H_T_H, I_STATE: 18x18 矩阵
    G = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
    H_T_H = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
    I_STATE = torch.eye(DIM_STATE, dtype=DOUBLE, device=DEVICE)
    
    # rot_add, t_add: 3D 向量 (3x1)
    rot_add = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
    t_add = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
    state_propagat = StatesGroup()
    # pointOri = PointXYZINormal()
    # pointSel = PointXYZINormal()
    # coeff = PointXYZINormal()
    
    corr_normvect: List[PointXYZI] = []
    frame_num: int = 0
    
    deltaT: int = 0
    deltaR: int = 0
    aver_time_consu: int = 0
    flg_EKF_inited: bool = False
    flg_EKF_converged: bool = False
    EKF_stop_flg: bool = False
    is_first_frame: bool = True
    #
    downSizeFilterSurf = None
    #

    p_imu = ImuProcess()
    p_imu.imu_en = imu_en
    extT = extrinT.clone()
    extR = extrinR.clone()
    p_imu.set_extrinsic(transl=extT, rot=extR)

    print("use imu") if imu_en else print("no imu")
    
    p_imu.set_gyr_cov_scale(torch.tensor([[gyr_cov_scale], [gyr_cov_scale], [gyr_cov_scale]], dtype=DOUBLE, device=DEVICE))
    p_imu.set_acc_cov_scale(torch.tensor([[acc_cov_scale], [acc_cov_scale], [acc_cov_scale]], dtype=DOUBLE, device=DEVICE))
    p_imu.set_gyr_bias_cov(torch.tensor([[0.00001], [0.00001], [0.00001]], dtype=DOUBLE, device=DEVICE))
    p_imu.set_acc_bias_cov(torch.tensor([[0.00001], [0.00001], [0.00001]], dtype=DOUBLE, device=DEVICE))

    init_map: bool = False
    voxel_map: Dict[VOXEL_LOC, OctoTree] = {}
    state = state_propagat
    #
    # if (flg_EKF_inited && !init_map) start
    #
    # 将点云转换到世界坐标系
    world_lidar = vx.transform_lidar(state, feats_undistort)  # 列表形式
    #
    # "for" change to "Batch" start
    #
    # 提取 feats_undistort 的点 (LiDAR 坐标系)
    points_this = torch.tensor(
        [[p.x, p.y, p.z] for p in feats_undistort], dtype=DOUBLE, device=DEVICE
    )  # 形状 (N, 3)

    # 提取 world_lidar 的点 (世界坐标系)
    points_world = torch.tensor(
        [[p.x, p.y, p.z] for p in world_lidar], dtype=DOUBLE, device=DEVICE
    )  # 形状 (N, 3)

    # 如果 z=0，设置为 0.001
    points_this[:, 2] = torch.where(points_this[:, 2] == 0, 0.001, points_this[:, 2])
    
    # 计算协方差
    covs = vx.calcBodyCov(points_this, ranging_cov, angle_cov)  # 形状 (N, 3, 3)
    # 协方差传播到世界坐标系
    points_this = points_this + Lidar_offset_to_IMU  # 形状 (N, 3)
    
    # 计算叉积矩阵
    point_crossmat = torch.zeros(points_this.shape[0], 3, 3, dtype=DOUBLE, device=DEVICE)
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

    vx.buildVoxelMap(pv_list, max_voxel_size, max_layer, max_cov_points_size,
                max_points_size, max_points_size, min_eigen_value, voxel_map)
    #
    # if (flg_EKF_inited && !init_map) end
    #
if __name__ == '__main__':
    args = read_yaml("config/cloud2voxel_mapping.yaml")
    print(args)
    main(args)
    
    # # 测试用例
    # feats_undistort = [PointXYZI(-9.10841, -6.09084, -1.2345, 0.0), PointXYZI(-9.10841, -6.09084, -1.2345, 0.0)]
    # state = StatesGroup()
    # range_inc = 0.04
    # degree_inc = 0.1
    # Lidar_offset_to_IMU = torch.tensor([[0, 0, 0]], dtype=DOUBLE, device="cuda")
    # max_layer = 2
    # max_voxel_size = 2
    # max_points_size = 1000
    # max_cov_points_size = 1000
    # min_eigen_value = 0.01
    # voxel_map: Dict[VOXEL_LOC, OctoTree] = {}
    # c2v(state, feats_undistort, range_inc, degree_inc, Lidar_offset_to_IMU,
    #     max_layer, max_voxel_size, max_points_size, max_cov_points_size, min_eigen_value, voxel_map)