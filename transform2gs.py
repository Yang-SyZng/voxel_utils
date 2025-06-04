from main import cloud2voxel, read_yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import torch
from typing import Final, List, Dict
from voxel_utils.utils.voxel_map_util import VOXEL_LOC, OctoTree

args = read_yaml("config/cloud2voxel_mapping.yaml")
voxel_map = cloud2voxel(args)

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
def traverse_octo_tree_bfs(voxel_map: Dict[VOXEL_LOC, OctoTree]) -> List[OctoTree]:
        """
        遍历 voxel_map 中的每个 OctoTree 并执行广度优先遍历（BFS）。
        保存所有不为 None 的 OctoTree 和 OctoTree 叶子节点。
        返回一个列表，包含所有有效节点。
        """
        valid_nodes = []  # 用于保存所有不为 None 的 OctoTree 和叶子节点
        
        # 遍历 voxel_map 中每一个 voxel 和它对应的 OctoTree
        for voxel_loc, octo_tree in voxel_map.items():

            queue = [octo_tree]  # 初始化队列，加入当前的 OctoTree
            while queue:
                node = queue.pop(0)  # 从队列的前端移除一个节点
                # 如果节点是叶子节点，输出信息并添加到 valid_nodes 列表
                if node.octo_state_ == 0:
                    valid_nodes.append(node)  # 添加叶子节点到 valid_nodes
                else:
                    # 否则，遍历当前节点的子节点
                    for i, leaf in enumerate(node.leaves_):
                        if leaf is not None:
                            queue.append(leaf)  # 将子节点加入队列

        return valid_nodes
valid_nodes = traverse_octo_tree_bfs(voxel_map)
# 收集顶点数据，并剔除包含 NaN 的数据
vertices = []
for node in valid_nodes:
    center = node.plane_ptr_.center.clone().cpu().numpy()
    radius = node.plane_ptr_.radius
    normal = node.plane_ptr_.normal.clone().cpu().numpy()
    normal = np.asarray(normal).reshape(-1)
    # 归一化法向量
    norm = np.linalg.norm(normal)
    if norm > 0:  # 避免除以零
        normal = normal / norm
    else:
        continue  # 如果法向量无效，跳过
    rotation = np.asarray([node.plane_ptr_.x_normal.clone().cpu().numpy(), node.plane_ptr_.y_normal.clone().cpu().numpy(), node.plane_ptr_.normal.clone().cpu().numpy()])
    quat = rotation_matrix_to_quaternion(rotation)
    # 颜色 (红色示例)
    f_dc = [1.0, 0.0, 0.0]
    f_rest = [0.0] * 45  # 高阶 SH 填充 0
    opacity = 1.0
    # scales = [-1.5*radius, -1.5*radius, -10]  # 薄片
    scales = [np.log(radius), np.log(radius), -10]  # 薄片
    # 构造顶点数据
    vertex = (
        center[0], center[1], center[2],  # x, y, z
        normal[0], normal[1], normal[2],  # nx, ny, nz
        # 0, 0, 0,
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
print(len(vertex_array))
# 写入 PLY 文件
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

# 写入二进制文件
with open("output/point_cloud/point_cloud.ply", "wb") as f:
    # 写入头部（ASCII 格式）
    f.write(header.encode('ascii'))
    # 写入二进制顶点数据
    vertex_array.tofile(f)


# with open("./origin_point_cloud.ply", "w") as f:
#     # 写入头部（直接写入 ASCII 格式）
#     f.write(header)
#     # 写入顶点数据（需要将 vertex_array 转换为 ASCII 格式）
#     for vertex in vertex_array:
#         f.write(" ".join(map(str, vertex)) + "\n")