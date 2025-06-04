import argparse
from argparse import Namespace
import yaml
import torch
from typing import Final, List, Dict
import voxel_utils.lib.common_lib as cl
from voxel_utils.lib.common_lib import StatesGroup, ImuProcess, \
                            PointXYZINormal, PointXYZI, \
                            MeasureGroup, Lidar_offset_to_IMU, \
                            TimestampUpdater # PointXYZINormal
from voxel_utils.lib import DIM_STATE
from voxel_utils.utils.voxel_map_util import pointWithCov, VOXEL_LOC, OctoTree, Ptpls
from voxel_utils.utils import DOUBLE, DEVICE
import voxel_utils.utils.voxel_map_util as vx
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
solution = torch.zeros(DIM_STATE, dtype=DOUBLE, device=DEVICE)
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
laserCloudOri = PointXYZINormal()
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
        if (points_tensor.shape != normals_tensor.shape and normals_tensor.shape != (0, 3)) or points_tensor.shape[1] != 3:
            raise ValueError("points_tensor and normals_tensor must have shape (N, 3) and match in size")
        meas.lidar.points = points_tensor
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

def pointBodyToWorld(pi: torch.Tensor):
    # pi shape (N, 3)
    p_body = pi + Lidar_offset_to_IMU.view(1, 3)
    # (3, 3) @ (N, 3) + (1, 3) -> (N, 3)
    p_global = p_body @ state.rot_end.T + state.pos_end.view(1, 3)
    return p_global

def RotMtoEuler(rot):
    assert rot.shape == (3, 3), f"Expected rot shape (3, 3), got {rot.shape}"
    sy = torch.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(rot[2, 1], rot[2, 2])
        y = torch.atan2(-rot[2, 0], sy)
        z = torch.atan2(rot[1, 0], rot[0, 0])
    else:
        x = torch.atan2(-rot[1, 2], rot[1, 1])
        y = torch.atan2(-rot[2, 0], sy)
        z = torch.tensor(0.0, device=DEVICE, dtype=DOUBLE)

    return torch.tensor([x, y, z], device=DEVICE, dtype=DOUBLE)

def cloud2voxel(args: Namespace, input_pcd=None):
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
    
    if input_pcd is None:
        pcd = readPointCloud(file_path, file_format)
        points_tensor = torch.tensor(np.asarray(pcd.points), dtype=DOUBLE, device=DEVICE)
        normals_tensor = torch.tensor(np.asarray(pcd.normals), dtype=DOUBLE, device=DEVICE)
        rgb_tensor = torch.tensor(np.asarray(pcd.colors), dtype=DOUBLE, device=DEVICE)
    else:
        pcd = input_pcd
        points_tensor = torch.tensor(pcd.points, dtype=DOUBLE, device=DEVICE)
        normals_tensor = torch.tensor(pcd.normals, dtype=DOUBLE, device=DEVICE)
        rgb_tensor = torch.tensor(pcd.colors, dtype=DOUBLE, device=DEVICE)
    
    
    # solution: 18x1 列向量
    solution = torch.zeros(DIM_STATE, dtype=DOUBLE, device=DEVICE)

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
    while scanIdx < 2:
        print(f"scanIdx:{scanIdx}")
        if sync_packages(Measures):
            match_time = 0
            solve_time = 0
            svd_time = 0
            state, feats_undistort = p_imu.Process(Measures, state)
        
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
                points_this = feats_undistort.points.clone()  # 形状 (N, 3)

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
                pv_list = pointWithCov(points=points_world, covs=covs)

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

                # if publish_voxel_map:
                #     vx.pubVoxelMap(voxel_map, publish_max_voxel_layer)
                    
                init_map = True
                scanIdx += 1
                
                # print("编译通过！")
                # exit(-1)
                continue
            
            rematch_num = 0
            nearest_search_en: bool = True
            
            #
            # downsample the feature points in a scan
            #
            # print(state.cov)
            # t_downsample_start = time.perf_counter()
            feats_down_body = vx.downsample_point_cloud(feats_undistort, voxel_size=filter_size_surf_min)
            # t_downsample_end = time.perf_counter()
            print(f"feats size: {feats_undistort.size}, down size: {feats_down_body.size}")
            # t_downsample = (t_downsample_end - t_downsample_start) * 1000  # 转换为毫秒
            
            # calc_point_cov_start = time.perf_counter()
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
            # calc_point_cov_end = time.perf_counter()


            for iterCount in range(NUM_MAX_ITERATIONS-1):
                # 初始化
                laserCloudOri.clear
                laserCloudNoeffect.clear
                corr_normvect.clear

                total_residual: float = 0.0

                # r_list = []
                ptpl_list = []

                # 转换LiDAR
                # 假设 transformLidar 已自定义或存在
                world_lidar = vx.transformLidar(state, feats_down_body)

                pv_list = pointWithCov()
                
                pv = pointWithCov(points=feats_down_body.points)
                pv.update_point_world(world_lidar.points)
                cov = body_var.clone()
                point_crossmat = crossmat_list.clone()
                rot_var = state.cov[:3, :3]
                t_var = state.cov[3:6, 3:6]
                # (3, 3) * (3, 3) * (3, 3)^T + (6051, 3, 3) * (3, 3) * (6051, 3, 3)^T + (3, 3)
                cov = state.rot_end * cov * state.rot_end.T + \
                        (-point_crossmat) * rot_var.unsqueeze(0) * (-point_crossmat.transpose(-2, -1)) + \
                        t_var
                pv.covs = cov
                pv_list = pv
                var_list = cov
                
                # BuildResidualListOMP 已定义
                ptpl_list = vx.buildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list)

                effct_feat_num = 0
                total_residual = 0.0

                ptpls = Ptpls()
                for i in range(len(ptpl_list)):
                    ptpls.add_data(point=ptpl_list[i].point, normal=ptpl_list[i].normal,
                                   center=ptpl_list[i].center, plane_cov=ptpl_list[i].plane_cov,
                                   d=ptpl_list[i].d, layer=ptpl_list[i].layer)
                effct_feat_num = ptpls.points.shape[0]
                pi_body = ptpls.points
                pi_world = pointBodyToWorld(pi_body)  # 自定义实现
                
                pl = ptpls.normals
                # (N, 3) (N, 3) + (N, 1) -> (N, 1)
                dis = torch.sum(pi_world * pl, dim=1, keepdim=True) + ptpls.ds
                # total_residual += abs(dis)

                # 保存到对应容器
                laserCloudOri.add_points(points=pi_body)
                
                # 处理corr_normvect（法向量）等
                corr_normvect = PointXYZI(points=pl, intensity=dis)
                # print(pl.shape, dis.shape)
                # res_mean_last = total_residual / effct_feat_num if effct_feat_num != 0 else 0
                # 开始时间
                # t_solve_start = time.time()
                # 计算Jacobian和测量向量
                Hsub = torch.zeros(effct_feat_num, 6, dtype=DOUBLE, device=DEVICE)
                Hsub_T_R_inv = torch.zeros(6, effct_feat_num, dtype=DOUBLE, device=DEVICE)
                R_inv = torch.zeros(effct_feat_num, dtype=DOUBLE, device=DEVICE)
                meas_vec = torch.zeros(effct_feat_num, dtype=DOUBLE, device=DEVICE)
                point_this = laserCloudOri.points
                
                if calib_laser:
                    covs = vx.calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV)
                else:
                    covs = vx.calcBodyCov(point_this, ranging_cov, angle_cov)
                # (3, 3) @ (N, 3, 3) * (3, 3) -> (N, 3, 3)
                covs = state.rot_end @ covs @ state.rot_end.T
                # (N, 3, 3)
                crossmat_list_temp = torch.zeros((point_this.shape[0], 3, 3), dtype=DOUBLE, device=DEVICE)
                crossmat_list_temp[:, 0, 1] = -point_this[:, 2]
                crossmat_list_temp[:, 0, 2] = point_this[:, 1]
                crossmat_list_temp[:, 1, 0] = point_this[:, 2]
                crossmat_list_temp[:, 1, 2] = -point_this[:, 0]
                crossmat_list_temp[:, 2, 0] = -point_this[:, 1]
                crossmat_list_temp[:, 2, 1] = point_this[:, 0]  # 形状 (N, 3, 3)
                
                norm_p = corr_normvect
                norm_vec = norm_p.points
                # (3, 3) @ (N, 3)  + (3) -> (N, 3)
                point_world = point_this @ state.rot_end + state.pos_end
                # 计算J_nq
                J_nq = torch.zeros(point_this.shape[0], 1, 6, dtype=DOUBLE, device=DEVICE)
                # print(J_nq.shape, point_world.shape)
                J_nq[:, :1, :3] = (point_world + ptpls.centers).unsqueeze(1)
                J_nq[:, :1, 3:6] = -ptpls.normals.unsqueeze(1)
                # (N, 1, 6) (N, 6, 6) (N, 6, 1) -> (N, 1)
                temp = torch.bmm(J_nq, ptpls.plane_covs)
                sigma_l = torch.bmm(temp, J_nq.transpose(2, 1)).squeeze(-1)
                # 1 / (N, 1) + (N, 3) (N, 3, 3) (N, 3)
                norm_vec_col = norm_vec.unsqueeze(-1)  # (N, 3, 1)
                norm_vec_row = norm_vec_col.transpose(1, 2)  # (N, 1, 3)
                proj_var = torch.bmm(torch.bmm(norm_vec_row, covs), norm_vec_col).squeeze(-1)  # (N, 1)
                R_inv = 1.0 / (sigma_l + proj_var)
                laserCloudOri.update_intensity(torch.sqrt(R_inv))
                # (N, 1) (N, 1) (N, 1) -> (N, 3)
                normal = torch.cat([corr_normvect.intensity, torch.sqrt(sigma_l),
                                    torch.sqrt(proj_var)], dim=1)
                laserCloudOri.update_normals(normal)
                laserCloudOri.update_curvature(torch.sqrt(sigma_l + proj_var))
                
                
                # 计算Jacobian H
                # (N, 3, 3) (3, 3) (N, 3)
                rotated_norm = state.rot_end.T @ norm_vec.unsqueeze(-1)  # (N, 3, 1)
                A = (crossmat_list_temp @ rotated_norm).squeeze(-1)  # (N, 3)
                
                Hsub[:, :3] = A
                Hsub[:, 3:] = norm_p.points
                # [(N, 3) (N, 1) -> (N, 3)  (N, 3) (N, 1) -> (N, 3) ] -> (6, N)
                Hsub_T_R_inv = torch.cat([A * R_inv, norm_p.points * R_inv], dim=1).T

                meas_vec = -norm_p.intensity
                
                # 计算核
                # 根据是否初始化和迭代更新
                if not True:
                    pass
                else:
                    Hsub_T = Hsub.T
                    # (6, N) (N, 6) -> (6,6)
                    H_T_H[:6, :6] = Hsub_T_R_inv @ Hsub
                    k_1 = (H_T_H + state.cov.inverse()).inverse()
                    # (18, 6) (6, N)
                    K = k_1[:, :6] @ Hsub_T_R_inv
                    vec = (state_propagat - state).unsqueeze(1)
                    # (18, N) (N, 1) (18, 1) - (18, N) (N, 6) (6, 1) -> (18, 1)
                    
                    solution = (K @ meas_vec + vec - K @ Hsub @ vec[:6]).squeeze(1)
                    state += solution
                    # 判断是否收敛
                    rot_add = solution[:3]
                    t_add = solution[3:6]

                    # 计算角度和位置变化
                    deltaR = torch.norm(rot_add) * 57.3
                    deltaT = torch.norm(t_add) * 100

                    if deltaR < 0.01 and deltaT < 0.015:
                        flg_EKF_converged = True

                # euler_cur = RotMtoEuler(state.rot_end)
                if flg_EKF_converged or ((rematch_num == 0) and (iterCount == (NUM_MAX_ITERATIONS - 2))):
                    nearest_search_en = True
                    rematch_num += 1
                if EKF_stop_flg and (rematch_num >= 2 or (iterCount == (NUM_MAX_ITERATIONS - 1))):
                    # if flg_EKF_inited:
                    if True:
                        G = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)
                        # (18, N) (N, 6) -> 18,16
                        G[:, :6] = K @ Hsub
                        
                        state.cov = (I_STATE - G) * state.cov

                        # K_sum = K.sum(dim=1)
                        # P_diag = state.cov.diag()
                    EKF_stop_flg =True
                
                if EKF_stop_flg:
                    break
            
            # add the  points to the voxel map
            world_lidar = PointXYZI()
            world_lidar = vx.transformLidar(state, feats_down_body)
            pv_list = pointWithCov()
            pv_list.points = feats_down_body.points
            pv_list.point_world = world_lidar.points
            cov = body_var
            point_crossmat = crossmat_list
            rot_var = state.cov[:3, :3]
            t_var = state.cov[3: 6, 3: 6]
            covs = state.rot_end * cov * state.rot_end.T + \
                        (-point_crossmat) * rot_var.unsqueeze(0) * (-point_crossmat.transpose(-2, -1)) + \
                        t_var
            pv_list.covs = covs
            
            vx.updateVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                             max_points_size, max_points_size, min_eigen_value,
                             voxel_map)
            
            
            scanIdx += 1
    print("done!")
    print("size ", len(voxel_map))
    return voxel_map
        
            #
            # if (flg_EKF_inited && !init_map) end
            #
            # test again
if __name__ == '__main__':
    args = read_yaml("config/cloud2voxel_mapping.yaml")
    # print(args)
    voxel_map = cloud2voxel(args)
    voxel_num = 0
    for _, value in voxel_map.items():
        if value.octo_state_ == 0:
            voxel_num += 1
        else:
            for leaf_value in value.leaves_:
                if leaf_value is not None:
                    if leaf_value.octo_state_ == 0:
                        voxel_num += 8
                    else:
                        for leaf_leaf_value in leaf_value.leaves_:
                            if leaf_leaf_value is not None:
                                voxel_num += 8
                        else:
                            continue
                else:
                    continue
    print(voxel_num)