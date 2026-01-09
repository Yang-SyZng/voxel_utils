import numpy as np
from voxel_utils import VoxelMap, OctoTree, Plane
from tqdm import tqdm
from scene.dataset_readers import fetchPly
from types import SimpleNamespace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
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


def visualize_planes_mpl(voxel_map_first, alpha=0.5):
    """
    批量显示矩形面片
    :param planes_list: 包含 Plane 对象的列表，每个对象需有 center, rotation, scale 属性
    :param alpha: 透明度
    """
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111, projection='3d')
    
    all_verts = []
    
    for node in voxel_map_first:
        try:
            c = node.plane_ptr_.center
        except AttributeError:
            continue
        # 1. 获取参数
        c = node.plane_ptr_.center
        R = node.plane_ptr_.rotation
        # 假设 scale 是 [width, height]
        sw, sh = node.plane_ptr_.scale0, node.plane_ptr_.scale1
        
        # 2. 定义局部坐标系下的 4 个顶点 (XY平面)
        local_verts = np.array([
            [-sw, -sh, 0],
            [ sw, -sh, 0],
            [ sw,  sh, 0],
            [-sw,  sh, 0]
        ])
        # local_verts = np.array([
        #     [-sw/2, -sh/2, 0],
        #     [ sw/2, -sh/2, 0],
        #     [ sw/2,  sh/2, 0],
        #     [-sw/2,  sh/2, 0]
        # ])
        
        # 3. 变换到世界坐标系: P_world = R * P_local + Center
        # 注意：R 是 3x3，local_verts 是 4x3，需要转置进行矩阵乘法
        world_verts = (R @ local_verts.T).T + c
        all_verts.append(world_verts)

    # 4. 创建 Poly3DCollection 批量添加
    # facecolors 可以传入列表来为每个面片设置不同颜色
    poly_collection = Poly3DCollection(all_verts, alpha=alpha, facecolors='cyan', edgecolors='b', linewidths=0.5)
    
    ax.add_collection3d(poly_collection)

    # 5. 设置坐标轴范围 (Matplotlib 3D 需手动调整范围以正常显示)
    all_points = np.vstack(all_verts)
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.axis('off')
    # 1. 关闭网格线
    ax.grid(False) 

    # 2. 隐藏背景面板（透明化 3D 盒子的三个面）
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 3. 隐藏坐标轴线（但保留标签，可选）
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

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
    visualize_planes_mpl(voxel_map_first)

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