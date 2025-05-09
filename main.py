import argparse
from argparse import Namespace
import yaml
import torch
from typing import Final, List, Dict
import lib.common_lib as cl
from lib.common_lib import StatesGroup, ImuProcess, PointCloudXYZINormal, PointCloudXYZI, MeasureGroup, Lidar_offset_to_IMU # PointXYZINormal
from lib import DIM_STATE
from utils.voxel_map_util import pointWithCov, VOXEL_LOC, OctoTree
from utils import DOUBLE, DEVICE
import utils.voxel_map_util as vx
import open3d as o3d
import numpy as np
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

feats_undistort = PointCloudXYZINormal()
feats_down_body = PointCloudXYZI()
laserCloudOri = PointCloudXYZI()
laserCloudNoeffect = PointCloudXYZI()
lidar_buffer = PointCloudXYZINormal()

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

def main(*args: Namespace):
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
    extrinT = torch.tensor(args.extrinsic_T, dtype=DOUBLE, device=DEVICE).reshape(3, 1)
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
    rot_add = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
    t_add = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
    state_propagat = StatesGroup()
    state = StatesGroup()
    # pointOri = PointXYZINormal()
    # pointSel = PointXYZINormal()
    # coeff = PointXYZINormal()
    
    corr_normvect = []
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
    
    last_rot = torch.eye(3, dtype=DOUBLE, device=DEVICE)

    Measures = MeasureGroup()
    def sync_packages(meas: MeasureGroup):
        if not imu_en:
            # 确保输入张量形状匹配
            if points_tensor.shape != normals_tensor.shape or points_tensor.shape[1] != 3:
                raise ValueError("points_tensor and normals_tensor must have shape (N, 3) and match in size")
            # 构造 intensity 列
            intensity = torch.zeros((points_tensor.shape[0], 1), dtype=DOUBLE, device=DEVICE)
            curvature = torch.zeros((points_tensor.shape[0], 1), dtype=DOUBLE, device=DEVICE)
            # 合并 points_tensor, intensity 和 normals_tensor 成形状 (N, 7)
            combined_tensor = torch.cat([
                points_tensor,  # (N, 3) -> [x, y, z]
                intensity,      # (N, 1) -> [intensity]
                normals_tensor, # (N, 3) -> [nx, ny, nz]
                curvature       # (N, 1) -> [curvature]
            ], dim=1)  # 结果形状 (N, 7)
            # 一次性添加到 meas.lidar
            meas.lidar.add_points(combined_tensor)
        return meas

    Measures = sync_packages(Measures)
    
    #
    # while
    #
    match_time = 0
    solve_time = 0
    svd_time = 0
    # for row in state.cov:
    #     print(' '.join(f'{v:.0e}' if v != 0 else '    0' for v in row))
    state, feats_undistort = p_imu.Process(Measures, state)
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
        # q?
        # 将点云转换到世界坐标系
        
        world_lidar = vx.transformLidar(state, feats_undistort)  # 列表形式
        
        #
        # "for" change to "Batch" start
        #
        # 提取 feats_undistort 的点 (LiDAR 坐标系)
        points_this = feats_undistort.points[:, :3]  # 形状 (N, 3)

        # 提取 world_lidar 的点 (世界坐标系)
        points_world = world_lidar.points[:, :3] # 形状 (N, 3)
        # 如果 z=0，设置为 0.001
        points_this[:, 2] = torch.where(points_this[:, 2] == 0, 0.001, points_this[:, 2])
        # 计算协方差
        covs = vx.calcBodyCov(points_this, ranging_cov, angle_cov)  # 形状 (N, 3, 3)
        
        # 协方差传播到世界坐标系
        points_this = points_this + Lidar_offset_to_IMU.T  # 形状 (N, 3)
        
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
        pv_list.add_points(points_world, covs)

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

        init_map = True
        
        circles = []
        filled_circles = []  # 用于存储填充的圆

        num_circle_points = 50  # 圆的分辨率

        for value in voxel_map.values():
            center = value.plane_ptr_.center.clone().cpu().numpy()
            radius = value.plane_ptr_.radius
            normal = value.plane_ptr_.normal.clone().cpu().numpy()
            normal = np.asarray(normal).reshape(-1)  # 确保是 (3,)
            
            # 生成圆的点 (在 XY 平面)
            theta = np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False)  # 不包含 2π 以避免重复
            circle_points = np.stack([
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.zeros_like(theta)
            ], axis=1)

            # 计算旋转矩阵: 从 (0, 0, 1) -> normal
            z_axis = np.array([0, 0, 1])
            normal = normal / np.linalg.norm(normal)

            v = np.cross(z_axis, normal)
            c = np.dot(z_axis, normal)
            if np.allclose(v, 0):  # 平行或反向时
                if c > 0:
                    R = np.eye(3)
                else:
                    R = -np.eye(3)
            else:
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
                R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

            # --- 创建填充的圆 (TriangleMesh) ---
            # 添加圆心点
            vertices = np.vstack([circle_points, [0, 0, 0]])  # 最后一个点是圆心
            num_vertices = len(vertices)

            # 生成三角面（扇形三角化）
            triangles = []
            center_idx = num_vertices - 1  # 圆心的索引
            for i in range(num_circle_points):
                next_i = (i + 1) % num_circle_points
                triangles.append([i, next_i, center_idx])

            # 旋转到指定平面
            rotated_vertices = vertices @ R.T

            # 平移到 center
            rotated_vertices += center

            # 创建 TriangleMesh
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(rotated_vertices),
                triangles=o3d.utility.Vector3iVector(triangles)
            )

            # 设置颜色（例如红色）
            mesh.paint_uniform_color([1, 0, 0])  # RGB 红色

            # 计算三角面法向量（确保显示正确）
            mesh.compute_vertex_normals()

            filled_circles.append(mesh)

            # --- 原有的 LineSet（可选，保留线框） ---
            # 旋转到指定平面
            rotated_circle = circle_points @ R.T

            # 平移到 center
            rotated_circle += center

            # 创建 line 对应的 index (闭合)
            lines = [[i, (i + 1) % num_circle_points] for i in range(num_circle_points)]

            # 创建 LineSet
            circle = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(rotated_circle),
                lines=o3d.utility.Vector2iVector(lines)
            )

            # 设置颜色（比如红色）
            circle.paint_uniform_color([1, 0, 0])  # 使用 paint_uniform_color 统一颜色

            circles.append(circle)

        # 确保点的数量
        num_points = np.asarray(pcd.points).shape[0]

        # 设置灰色 (比如 0.2, 0.2, 0.2)
        gray_color = np.tile([0.2, 0.2, 0.2], (num_points, 1))

        # 赋值
        pcd.colors = o3d.utility.Vector3dVector(gray_color)

        # 可视化（显示点云、线框和填充圆）
        o3d.visualization.draw_geometries([pcd, *circles, *filled_circles])
        
        # # 收集 centers
        # centers = []
        # radiuses = []
        # for _, value in voxel_map.items():
        #     center = value.plane_ptr_.center  # 假设是 list/array-like
        #     radius = value.plane_ptr_.radius
        #     normal = value.plane_ptr_.normal
        #     centers.append(center.clone().cpu().numpy())
        #     radiuses.append(radius)
            
        # # 转换为 numpy 数组
        # centers_np = np.array(centers)

        # # 创建 Open3D 点云
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(centers_np)
        # # 设置 pcd1 的颜色为红色
        # num_points_pcd1 = centers_np.shape[0]
        # red_color = np.tile([1, 0, 0], (num_points_pcd1, 1))
        # pcd1.colors = o3d.utility.Vector3dVector(red_color)

        # # 确保点的数量
        # num_points = np.asarray(pcd.points).shape[0]

        # # 设置灰色 (比如 0.5, 0.5, 0.5)
        # gray_color = np.tile([0.2, 0.2, 0.2], (num_points, 1))

        # # 赋值
        # pcd.colors = o3d.utility.Vector3dVector(gray_color)
        
        # # 可视化
        # o3d.visualization.draw_geometries([pcd1, pcd])
        # # 保存为 .ply 文件
        # o3d.io.write_point_cloud("centers_output.ply", pcd1)
        #
        # if (flg_EKF_inited && !init_map) end
        #
    
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