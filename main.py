import argparse
from argparse import Namespace
import yaml
import torch
from typing import Final, List, Dict
import lib.common_lib as cl
from lib.common_lib import StatesGroup, ImuProcess, \
                            PointXYZINormal, PointXYZI, \
                            MeasureGroup, Lidar_offset_to_IMU, \
                            TimestampUpdater # PointXYZINormal
from lib import DIM_STATE
from utils.voxel_map_util import pointWithCov, VOXEL_LOC, OctoTree
from utils import DOUBLE, DEVICE
import utils.voxel_map_util as vx
import open3d as o3d
import numpy as np
import time
# torch.set_printoptions(precision=5, linewidth=1000)
torch.set_printoptions(sci_mode=False, precision=12, linewidth=1000)
#  VV \        VV \   AAAAAAAA\    LL\          UU\     UU\   EEEEEEEEEEE\  SSSSSSSS\
#   VV \      VV /   AA  ____AA\   LL |         UU |    UU |  EE  ______|  SS  ______|
#    VV \    VV /    AA /    AA |  LL |         UU |    UU |  EE |         SS /
#     VV \  VV /     AAAAAAAAAA |  LL |         UU |    UU |  EEEEEEEEEE\    SSSSSSS \
#      VV \VV /      AA  ____AA |  LL |         UU |    UU |  EE  ______|           SS \
#       VVVV /       AA |    AA |  LL |         UU |    UU |  EE |                  SS |
#        VV /        AA |    AA |  LLLLLLLLLL\   UUUUUUUU /   EEEEEEEEEEE\   SSSSSSSS /
#        \_/         \__|    \__|  \_________|   \_______/    \__________|   \_______/
# Created by zty 2025/05/07

INIT_TIME: Final = 0.0
CALIB_ANGLE_COV: Final = 0.01

scanIdx = 0
lid_topic = None
imu_topic = None
ranging_cov = None
angle_cov = None
gyr_cov_scale = None
acc_cov_scale = None
imu_en = None
extrinT = None
extrinR = None
NUM_MAX_ITERATIONS = None
max_points_size = None
max_cov_points_size = None
layer_point_size = None
layer_size = None
max_layer = None
max_voxel_size = None
filter_size_surf_min = None
min_eigen_value = None
calib_laser = None
scan_line = None
publish_voxel_map = None
publish_max_voxel_layer = None
publish_point_cloud = None
pub_point_cloud_skip = None
dense_map_en = None
write_kitti_log = None
result_path = None
file_path = None
file_format = None
pcd = None
points_tensor: torch.Tensor
normals_tensor: torch.Tensor
rgb_tensor = None
solution = torch.zeros((DIM_STATE, 1), dtype=DOUBLE, device=DEVICE)
G = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
H_T_H = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
I_STATE = torch.eye(DIM_STATE, dtype=DOUBLE, device=DEVICE)
rot_add = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
t_add = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
state_propagat = None
state: StatesGroup
corr_normvect = []
frame_num: int = 0
deltaT: int = 0
deltaR: int = 0
aver_time_consu: int = 0
flg_EKF_inited: bool  = False
flg_EKF_converged: bool = False
EKF_stop_flg: bool = False
is_first_frame: bool = True
downSizeFilterSurf = None

# time_buffer: float = 0.
last_timestamp_lidar: float = -1.0
timer: TimestampUpdater

Measures = MeasureGroup()

feats_undistort = PointXYZINormal()
feats_down_body = PointXYZI()
laserCloudOri = PointXYZI()
laserCloudNoeffect = PointXYZI()
lidar_buffer = PointXYZINormal()

# FFFFFFFF\    UU\     UU\    NN\    NN\     CCCCCCCCC\    TTTTTTTTTT\   IIIIII\      OOOOOOOO\      NN\     NN\     SSSSSSSS\
# FF  _____|   UU |    UU |   NNN\   NN |   CC ________|       TT  __|     II  _|    OO _____OO \    NNN\    NN |   SS  ______|
# FF |         UU |    UU |   NN NN  NN |   CC |               TT |        II |     OO /      OO |   NNNN\   NN |   SS /
# FFFFF\       UU |    UU |   NN \N\ NN |   CC |               TT |        II |     OO |      OO |   NN NN\  NN |    SSSSSSS \
# FF  __|      UU |    UU |   NN |\NNNN |   CC |               TT |        II |     OO |      OO |   NN | NN\NN |           SS \
# FF |         UU |    UU |   NN | \NNN |   CC |               TT |        II |      OO \    OO /    NN |  NNNN |           SS |
# FF |          UUUUUUUU /    NN |  \NN |    CCCCCCCCC\        TT |      IIIIII\      OOOOOOOO /     NN |   NNN |    SSSSSSSS /
# \__|          \_______/     \__|   \__|    \_________|       \__|      \______|     \_______|      \__|   \___|    \_______/
# Created by zty 2025/05/07



def read_yaml(yaml_path: str):
    """读取 YAML 配置文件，并转成 argparse.Namespace"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
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

def readPointCloud(file_path: str, file_format: str) -> o3d.geometry.PointCloud:
    if file_format not in ["pcd", "ply"]:
        raise ValueError(f"Unsupported file format: {file_format}")

    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        raise ValueError(f"Couldn't read {file_format.upper()} file {file_path}: {str(e)}")

    if not pcd.has_points():
        raise ValueError(f"Loaded point cloud is empty: {file_path}")

    return pcd

def sync_packages(meas: MeasureGroup):
    if not imu_en:
        # 确保输入张量形状匹配
        if points_tensor.shape != normals_tensor.shape or points_tensor.shape[1] != 3:
            raise ValueError("points_tensor and normals_tensor must have shape (N, 3) and match in size")
        meas.lidar.add_points(points=points_tensor)
        # meas.lidar_beg_time = time_buffer
        meas.lidar_beg_time = timer.timestamp.toSec()
        # print(f"sync_packages:{meas.lidar_beg_time}")
        return True
    return False

def standard_pcl_cbk(msg):
    global last_timestamp_lidar
    
    # 如果当前的时间戳小于上次的时间戳，说明是回环数据，清空缓冲区
    if msg['header']['stamp'] < last_timestamp_lidar:
        print("lidar loop back, clear buffer")
        # lidar_buffer.clear()

    # 假设这里是点云处理的过程，这里可以根据你的需求处理 msg
    # PointCloudXYZI 的处理逻辑在这里，简化成一个示例
    # ptr = {"points": msg['data']}  # 处理后的点云数据

    # 将点云数据和时间戳加入到队列中
    # lidar_buffer.append(ptr)
    time_buffer = timer.timestamp.toSec()
    
    # 更新上次的时间戳
    last_timestamp_lidar = time_buffer

    # 这里可以加上其他需要通知的操作（例如多线程通知等）
    # print(f"Received point cloud at time: {msg['header']['stamp']:.6f}")
    
def buildResidualListOMP(voxel_map, max_voxel_size, threshold, max_layer, pv_list, ptpl_list, non_match_list):
    # 构建残差列表
    pass

def pointBodyToWorld(pi: PointXYZINormal, po: PointXYZINormal):
    p_body = pi.points[:, :3] + Lidar_offset_to_IMU
    p_global = torch.matmul(state.rot_end, p_body) + state.pos_end
    # 构造 intensity 列
    intensity = torch.zeros((points_tensor.shape[0], 1), dtype=DOUBLE, device=DEVICE)
    curvature = torch.zeros((points_tensor.shape[0], 1), dtype=DOUBLE, device=DEVICE)
    # 合并 points_tensor, intensity 和 normals_tensor 成形状 (N, 7)
    combined_tensor = torch.cat([
        points_tensor,  # (N, 3) -> [x, y, z]
        intensity,      # (N, 1) -> [intensity]
        normals_tensor, # (N, 3) -> [nx, ny, nz]
        curvature       # (N, 1) -> [curvature]
    ], dim=1)  # 结果形状 (N, 8)
    po.add_points(combined_tensor)
    return po

def RotMtoEuler(rot_matrix):
    
    return np.zeros(3)

def main(*args: Namespace):
    global scanIdx, lid_topic, imu_topic, ranging_cov, angle_cov, gyr_cov_scale, acc_cov_scale
    global imu_en, extrinT, extrinR, NUM_MAX_ITERATIONS, max_points_size, max_cov_points_size
    global layer_point_size, layer_size, max_layer, max_voxel_size, filter_size_surf_min, min_eigen_value
    global calib_laser, scan_line, publish_voxel_map, publish_max_voxel_layer, publish_point_cloud
    global pub_point_cloud_skip, dense_map_en, write_kitti_log, result_path, file_path, file_format
    global pcd, points_tensor, normals_tensor, rgb_tensor
    global solution, G, H_T_H, I_STATE, rot_add, t_add, state_propagat, state
    global corr_normvect, frame_num, deltaT, deltaR, aver_time_consu
    global flg_EKF_inited, flg_EKF_converged, EKF_stop_flg, is_first_frame, downSizeFilterSurf
    global timer, Measures
    if isinstance(args, tuple):
        args = args[0]
    scanIdx = 0
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
    extrinT = torch.tensor(args.extrinsic_T, dtype=DOUBLE, device=DEVICE)
    extrinR = torch.tensor(args.extrinsic_R, dtype=DOUBLE, device=DEVICE).reshape(3, 3)
    
    # mapping algorithm params
    NUM_MAX_ITERATIONS = args.max_iteration
    max_points_size = args.max_points_size
    max_cov_points_size = args.max_cov_points_size
    layer_point_size = args.layer_point_size
    layer_size = layer_point_size
    max_layer = args.max_layer
    max_voxel_size = args.voxel_size
    filter_size_surf_min = args.down_sample_size
    min_eigen_value = args.plannar_threshold # min_eigen_value
    
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
    points_tensor = torch.tensor(np.asarray(pcd.points), dtype=DOUBLE, device=DEVICE)
    normals_tensor = torch.tensor(np.asarray(pcd.normals), dtype=DOUBLE, device=DEVICE)
    rgb_tensor = torch.tensor(np.asarray(pcd.colors), dtype=DOUBLE, device=DEVICE)
    
    # solution: 18x1 列向量
    solution = torch.zeros((DIM_STATE, 1), dtype=DOUBLE, device=DEVICE)

    # G, H_T_H, I_STATE: 18x18 矩阵
    G = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
    H_T_H = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
    I_STATE = torch.eye(DIM_STATE, dtype=DOUBLE, device=DEVICE)
    
    # rot_add, t_add: 3D 向量 (3x1)
    rot_add = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
    t_add = torch.zeros(3, dtype=DOUBLE, device=DEVICE)
    state_propagat = StatesGroup()
    state = StatesGroup()
    # pointOri = PointXYZINormal()
    # pointSel = PointXYZINormal()
    # coeff = PointXYZINormal()
    
    corr_normvect = []
    frame_num = 0
    
    deltaT = 0
    deltaR = 0
    aver_time_consu = 0
    flg_EKF_inited = False
    flg_EKF_converged = False
    EKF_stop_flg = False
    is_first_frame = True
    #
    downSizeFilterSurf = None
    #
    timer = TimestampUpdater(5.0)
    timer.start()
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
    
    # last_rot = torch.eye(3, dtype=DOUBLE, device=DEVICE)
    scanIdx = 0
    while True:
        print(f"scanIdx:{scanIdx}")
        if sync_packages(Measures):
            match_time = 0
            solve_time = 0
            svd_time = 0
            # for row in state.cov:
            #     print(' '.join(f'{v:.0e}' if v != 0 else '    0' for v in row))
            state, feats_undistort = p_imu.Process(Measures, state)
            # print(state.cov)
            # print(feats_undistort.points[:5])
            # for row in state.cov:
            #     print(' '.join(f'{v:.0e}' if v != 0 else '    0' for v in row))
        
            state_propagat = state
            
            if is_first_frame:
                first_lidar_time = Measures.lidar_beg_time
                is_first_frame = False
            if init_map == False: 
                #
                # if (flg_EKF_inited && !init_map) start
                #
                # 
                # 将点云转换到世界坐标系
                
                world_lidar = vx.transformLidar(state, feats_undistort)  # 列表形式
                
                #
                # "for" change to "Batch" start
                #
                # 提取 feats_undistort 的点 (LiDAR 坐标系)
                points_this = feats_undistort.points  # 形状 (N, 3)

                # 提取 world_lidar 的点 (世界坐标系)
                points_world = world_lidar.points # 形状 (N, 3)
                # 如果 z=0，设置为 0.001
                points_this[:, 2] = torch.where(points_this[:, 2] == 0, 0.001, points_this[:, 2])
                # 计算协方差
                covs = vx.calcBodyCov(points_this, ranging_cov, angle_cov)  # 形状 (N, 3, 3)
                
                # 协方差传播到世界坐标系
                points_this = points_this + Lidar_offset_to_IMU.view(3, 1).T  # 形状 (N, 3)
                
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
                pv_list = pointWithCov()
                pv_list.add_points(points=points_world, covs=covs)

                # 计算标准差
                sigma_pv = torch.diagonal(covs, dim1=1, dim2=2)  # 形状 (N, 3)
                sigma_pv = torch.sqrt(sigma_pv)  # 形状 (N, 3)
                #
                # "for" change to "Batch" end
                #
                # print("max_layer:", max_layer)
                # print(pv_list.points.shape)
                # print(max_voxel_size, max_layer, max_points_size, min_eigen_value)
                voxel_map = vx.buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                                            max_points_size, max_points_size, min_eigen_value,
                                            voxel_map)

                if publish_voxel_map:
                    vx.pubVoxelMap(voxel_map, publish_max_voxel_layer)
                    
                init_map = True
                scanIdx += 1
                
                print("编译通过！")
                exit(-1)
                continue
            
            
            #
            # downsample the feature points in a scan
            #
            # print(state.cov)
            t_downsample_start = time.perf_counter()
            feats_down_body = vx.downsample_point_cloud(feats_undistort, voxel_size=filter_size_surf_min)
            t_downsample_end = time.perf_counter()
            print(f"feats size: {feats_undistort.size}, down size: {feats_down_body.size}")
            t_downsample = (t_downsample_end - t_downsample_start) * 1000  # 转换为毫秒
            
            calc_point_cov_start = time.perf_counter()
            points_this = feats_down_body.points[:, :3]  # 形状 (N, 3)
            # 如果 z=0，设置为 0.001
            points_this[:, 2] = torch.where(points_this[:, 2] == 0, 0.001, points_this[:, 2])
            # 计算协方差
            covs = vx.calcBodyCov(points_this, ranging_cov, angle_cov)  # 形状 (N, 3, 3)
            # 计算叉积矩阵
            crossmat_list = torch.zeros(points_this.shape[0], 3, 3, dtype=DOUBLE, device=DEVICE)
            crossmat_list[:, 0, 1] = -points_this[:, 2]
            crossmat_list[:, 0, 2] = points_this[:, 1]
            crossmat_list[:, 1, 0] = points_this[:, 2]
            crossmat_list[:, 1, 2] = -points_this[:, 0]
            crossmat_list[:, 2, 0] = -points_this[:, 1]
            crossmat_list[:, 2, 1] = points_this[:, 0]  # 形状 (N, 3, 3)
        
            body_var = state.cov[3:6, 3:6].repeat(points_this.shape[0], 1, 1)
            calc_point_cov_end = time.perf_counter()


            for iterCount in range(NUM_MAX_ITERATIONS):
                # 初始化
                laserCloudOri = []
                laserCloudNoeffect = []
                corr_normvect = []
                total_residual: float = 0.0

                r_list = []
                ptpl_list = []

                # 转换LiDAR
                # 假设 transformLidar 已自定义或存在
                world_lidar = vx.transformLidar(state, p_imu, feats_down_body)

                pv_list = pointWithCov()
                
                world_lidar = vx.transformLidar(state, feats_down_body)
                pv = pointWithCov(points=feats_down_body.points[:, :3])
                pv.add_point_world(world_lidar.points[:, :3])
                cov = body_var.clone()
                point_crossmat = crossmat_list.clone()
                rot_var = state.cov[:, :3, :3]
                t_var = state.cov[:, 3:6, 3:6]
                # (3, 3) * (N, 3, 3) * (3, 3)^T + (N, 3, 3) * (3, 3) * (N, 3, 3)^T + (3, 3)
                cov = state.rot_end * cov * state.rot_end.T + \
                        (-point_crossmat) * rot_var * (-point_crossmat.T) + \
                        t_var
                pv.covs = cov
                pv_list = pv
                var_list = cov

                # 构建残差列表

                # 假设 BuildResidualListOMP 已定义
                ptpl_list, non_match_list = vx.BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list)

                effct_feat_num = 0
                total_residual = 0.0

                for i in range(len(ptpl_list)):
                    pi_body = ptpl_list[i]['point']
                    pi_world = pointBodyToWorld(pi_body, state)  # 自定义实现
                    pl = ptpl_list[i]['normal']

                    # 计算距离
                    dis = torch.dot(pi_world, pl) + ptpl_list[i]['d']
                    effct_feat_num += 1
                    total_residual += abs(dis)

                    # 保存到对应容器
                    laserCloudOri.append(pi_body)
                    # 处理corr_normvect（法向量）等
                    corr_normvect.append({'normal': pl, 'dis': dis})

                res_mean_last = total_residual / effct_feat_num if effct_feat_num != 0 else 0

                # 开始时间
                t_solve_start = time.time()

                # 计算Jacobian和测量向量
                Hsub = torch.zeros(effct_feat_num, 6)
                Hsub_T_R_inv = torch.zeros(6, effct_feat_num)
                R_inv = torch.zeros(effct_feat_num)
                meas_vec = torch.zeros(effct_feat_num)

                for i in range(effct_feat_num):
                    laser_p = laserCloudOri[i]
                    point_this = laser_p
                    # 例如 calcBodyCov
                    if calib_laser:
                        cov = vx.calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV)
                    else:
                        cov = vx.calcBodyCov(point_this, ranging_cov, angle_cov)

                    cov = state_rot_end @ cov @ state_rot_end.T
                    point_crossmat = crossmat_list[i]  # 需要提前定义
                    norm_p = corr_normvect[i]['normal']
                    norm_vec = norm_p
                    # 转换点到世界坐标
                    point_world = state_rot_end @ point_this + state_pos_end

                    # 计算J_nq
                    J_nq = torch.cat([point_world - ptpl_list[i]['center'], -ptpl_list[i]['normal']])
                    sigma_l = J_nq @ ptpl_list[i]['plane_cov'] @ J_nq.T
                    R_inv[i] = 1.0 / (sigma_l + norm_vec.T @ cov @ norm_vec)
                    dis = torch.norm(point_this)
                    # 赋值到点云
                    # 这里只是示意
                    # laserCloudOri[i].intensity = torch.sqrt(R_inv[i]) # 如果支持
                    # laserCloudOri[i].normal_x = corr_normvect[i]['normal'][0]等
                    # 改为pytorch tensor的操作

                    # 计算Jacobian H
                    A = point_crossmat @ (state_rot_end.T @ norm_vec)
                    Hsub[i, :3] = A
                    Hsub[i, 3:] = norm_p

                    Hsub_T_R_inv[:, i] = torch.cat([A * R_inv[i], norm_p * R_inv[i]])

                    meas_vec[i] = -dis

                # 计算核
                # 根据是否初始化和迭代更新
                if not flg_EKF_inited:
                    # 初始状态
                    H_init = torch.zeros(9, DIM_STATE)
                    z_init = torch.zeros(9, 1)
                    H_init[:3, :3] = torch.eye(3)
                    H_init[3:6, 3:6] = torch.eye(3)
                    H_init[6:, 12:] = torch.eye(3)  # 假设最后三维为位置

                    z_init[:3] = -Log(state_rot_end)
                    z_init[3:6] = -state_pos_end

                    K_init = state_cov @ H_init.T @ torch.inverse(H_init @ state_cov @ H_init.T + 0.0001 * torch.eye(9))
                    solution = K_init @ z_init

                    # 重置位置
                    state[:3] = torch.zeros(3)
                    state[3:6] = torch.zeros(3)
                    # 其他状态部分保持不变
                    EKF_stop_flg = True
                else:
                    # 计算卡尔曼增益
                    H_T_H = Hsub_T_R_inv @ Hsub
                    K = torch.inverse(H_T_H + torch.inverse(state_cov))
                    K = K @ Hsub_T_R_inv
                    vec = state_propagat - state
                    solution = K @ (meas_vec + vec - Hsub @ vec[:6])

                    # 判断是否收敛
                    rot_add = solution[:3]
                    t_add = solution[3:6]

                    state[:3] += rot_add
                    state[3:6] += t_add

                    # 计算角度和位置变化
                    deltaR = torch.norm(rot_add) * 57.3
                    deltaT = torch.norm(t_add) * 100

                    if deltaR < 0.01 and deltaT < 0.015:
                        flg_EKF_converged = True

                    # 更新协方差
                    G = torch.zeros(DIM_STATE, DIM_STATE)
                    G[:6, :6] = K @ Hsub
                    state_cov = (I_STATE - G) @ state_cov

                    total_distance += torch.norm(state[:3] - position_last)
                    position_last = state[:3]

                    # 更新四元数
                    euler_cur = RotMtoEuler(state[:3])  # 需要实现
                    geoQuat = tf.createQuaternionMsgFromRollPitchYaw(euler_cur[0], euler_cur[1], euler_cur[2])

                # 判断收敛，提前退出
                if flg_EKF_converged:
                    break

    # 其他的时间统计、日志等可以加上
                
            scanIdx += 1
            time.sleep(6)
            print(f"end")
            # exit(-1)
    return voxel_map
        
            #
            # if (flg_EKF_inited && !init_map) end
            #
            # test again
if __name__ == '__main__':
    args = read_yaml("config/cloud2voxel_mapping.yaml")
    # print(args)
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