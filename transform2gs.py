import numpy as np
from scipy.spatial.transform import Rotation as R
from voxel_utils import VoxelMap, OctoTree, Plane
from tqdm import tqdm
from scene.dataset_readers import fetchPly
from types import SimpleNamespace
# 定义 PLY 文件的顶点数据结构
dtype = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  # 位置
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # 法向量
    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),  # 颜色
    *[(f'f_rest_{i}', 'f4') for i in range(45)],  # 45 个 f_rest
    ('opacity', 'f4'),  # 透明度
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),  # 缩放
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')  # 四元数
]

def get_header(vertices):
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
""" + "\n".join([f"property float f_rest_{i}" for i in range(45)]) + """
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    return header

# 读取点云文件
def readPointCloud(file_path: str):
    try:
        pcd = fetchPly(file_path)
    except:
        pcd = None
    return pcd

# 旋转矩阵2四元数
def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]"""
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return [w, x, y, z]

def generate_ply(voxel_map_first):
    vertices = []
    for node in voxel_map_first:
        try:
            center = node.plane_ptr_.center
        except AttributeError:
            continue
        center = node.plane_ptr_.center
        normal = node.plane_ptr_.normal
        radius = node.plane_ptr_.radius
        # 归一化法向量
        norm = np.linalg.norm(normal)
        if norm > 0:  # 避免除以零
            normal = normal / norm
        else:
            continue  # 如果法向量无效，跳过
        rotation = node.plane_ptr_.rotation
        # rotation = np.asarray([node.plane_ptr_.x_normal, node.plane_ptr_.y_normal, node.plane_ptr_.normal])
        quat = rotation_matrix_to_quaternion(rotation)
        # 颜色 (红色示例)
        f_dc = [0.0, 0.0, 1.0]
        f_rest = [0.0] * 45  # 高阶 SH 填充 0
        opacity = 1.0
        # scales = [-1.5*radius, -1.5*radius, -10]  # 薄片
        scales = [np.log(radius), np.log(radius), -10]  # 薄片
        # 构造顶点数据
        vertex = (
            center[0], center[1], center[2],  # x, y, z
            normal[0], normal[1], normal[2],  # nx, ny, nz
            *f_dc,  # f_dc_0, f_dc_1, f_dc_2
            *f_rest,  # f_rest_0 到 f_rest_44
            opacity,  # opacity
            *scales,  # scale_0, scale_1, scale_2
            *quat  # rot_0, rot_1, rot_2, rot_3
        )
        # print(vertex)
        # 检查 vertex 中是否包含 NaN
        if not np.any(np.isnan(vertex)):
            vertices.append(vertex)
        else:
            print(f"Skipping vertex with NaN: center={center}, normal={normal}, quat={quat}")
    # 转换为 NumPy 结构化数组
    vertex_array = np.array(vertices, dtype=dtype)
    header = get_header(vertices)
    # 写入二进制文件
    with open("./point_cloud.ply", "wb") as f:
        # 写入头部（ASCII 格式）
        f.write(header.encode('ascii'))
        # 写入二进制顶点数据
        vertex_array.tofile(f)
        
def main(cfg, ply_path):
    pcd = readPointCloud(ply_path)
    voxel_map = VoxelMap(cfg=cfg, pcd=pcd)
    voxel_map_first = voxel_map.feat_map_first
    generate_ply(voxel_map_first=voxel_map_first)

if __name__ == "__main__":
    ply_path = "input.ply"

    # voxel_map parameters
    cfg_dict = {
        "voxel_size": 2.0,      # voxel_size
        "max_layer": 3,         # 4 layer, 0, 1, 2, 3
        "outliers_threshold": 5,# 离群点阈值
        "planar_threshold": 0.01# 点云高度阈值
    }
    cfg = SimpleNamespace(**cfg_dict)

    main(cfg, ply_path)