# circles = []
# filled_circles = []  # 用于存储填充的圆

# num_circle_points = 50  # 圆的分辨率

# for value in voxel_map.values():
#     center = value.plane_ptr_.center.clone().cpu().numpy()
#     radius = value.plane_ptr_.radius
#     normal = value.plane_ptr_.normal.clone().cpu().numpy()
#     normal = np.asarray(normal).reshape(-1)  # 确保是 (3,)
    
#     # 生成圆的点 (在 XY 平面)
#     theta = np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False)  # 不包含 2π 以避免重复
#     circle_points = np.stack([
#         radius * np.cos(theta),
#         radius * np.sin(theta),
#         np.zeros_like(theta)
#     ], axis=1)

#     # 计算旋转矩阵: 从 (0, 0, 1) -> normal
#     z_axis = np.array([0, 0, 1])
#     normal = normal / np.linalg.norm(normal)

#     v = np.cross(z_axis, normal)
#     c = np.dot(z_axis, normal)
#     if np.allclose(v, 0):  # 平行或反向时
#         if c > 0:
#             R = np.eye(3)
#         else:
#             R = -np.eye(3)
#     else:
#         s = np.linalg.norm(v)
#         kmat = np.array([[0, -v[2], v[1]],
#                         [v[2], 0, -v[0]],
#                         [-v[1], v[0], 0]])
#         R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

#     # --- 创建填充的圆 (TriangleMesh) ---
#     # 添加圆心点
#     vertices = np.vstack([circle_points, [0, 0, 0]])  # 最后一个点是圆心
#     num_vertices = len(vertices)

#     # 生成三角面（扇形三角化）
#     triangles = []
#     center_idx = num_vertices - 1  # 圆心的索引
#     for i in range(num_circle_points):
#         next_i = (i + 1) % num_circle_points
#         triangles.append([i, next_i, center_idx])

#     # 旋转到指定平面
#     rotated_vertices = vertices @ R.T

#     # 平移到 center
#     rotated_vertices += center

#     # 创建 TriangleMesh
#     mesh = o3d.geometry.TriangleMesh(
#         vertices=o3d.utility.Vector3dVector(rotated_vertices),
#         triangles=o3d.utility.Vector3iVector(triangles)
#     )

#     # 设置颜色（例如红色）
#     mesh.paint_uniform_color([1, 0, 0])  # RGB 红色

#     # 计算三角面法向量（确保显示正确）
#     mesh.compute_vertex_normals()

#     filled_circles.append(mesh)

#     # --- 原有的 LineSet（可选，保留线框） ---
#     # 旋转到指定平面
#     rotated_circle = circle_points @ R.T

#     # 平移到 center
#     rotated_circle += center

#     # 创建 line 对应的 index (闭合)
#     lines = [[i, (i + 1) % num_circle_points] for i in range(num_circle_points)]

#     # 创建 LineSet
#     circle = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(rotated_circle),
#         lines=o3d.utility.Vector2iVector(lines)
#     )

#     # 设置颜色（比如红色）
#     circle.paint_uniform_color([1, 0, 0])  # 使用 paint_uniform_color 统一颜色

#     circles.append(circle)

# # 确保点的数量
# num_points = np.asarray(pcd.points).shape[0]

# # 设置灰色 (比如 0.2, 0.2, 0.2)
# gray_color = np.tile([0.2, 0.2, 0.2], (num_points, 1))

# # 赋值
# pcd.colors = o3d.utility.Vector3dVector(gray_color)

# # 可视化（显示点云、线框和填充圆）
# o3d.visualization.draw_geometries([pcd, *circles, *filled_circles])

# # # 收集 centers
# # centers = []
# # radiuses = []
# # for _, value in voxel_map.items():
# #     center = value.plane_ptr_.center  # 假设是 list/array-like
# #     radius = value.plane_ptr_.radius
# #     normal = value.plane_ptr_.normal
# #     centers.append(center.clone().cpu().numpy())
# #     radiuses.append(radius)
    
# # # 转换为 numpy 数组
# # centers_np = np.array(centers)

# # # 创建 Open3D 点云
# # pcd1 = o3d.geometry.PointCloud()
# # pcd1.points = o3d.utility.Vector3dVector(centers_np)
# # # 设置 pcd1 的颜色为红色
# # num_points_pcd1 = centers_np.shape[0]
# # red_color = np.tile([1, 0, 0], (num_points_pcd1, 1))
# # pcd1.colors = o3d.utility.Vector3dVector(red_color)

# # # 确保点的数量
# # num_points = np.asarray(pcd.points).shape[0]

# # # 设置灰色 (比如 0.5, 0.5, 0.5)
# # gray_color = np.tile([0.2, 0.2, 0.2], (num_points, 1))

# # # 赋值
# # pcd.colors = o3d.utility.Vector3dVector(gray_color)

# # # 可视化
# # o3d.visualization.draw_geometries([pcd1, pcd])
# # # 保存为 .ply 文件
# # o3d.io.write_point_cloud("centers_output.ply", pcd1)

from main import main, read_yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

args = read_yaml("config/cloud2voxel_mapping.yaml")
voxel_map = main(args)

def normal_to_quaternion(normal):
    """从 (0,0,1) 旋转到 normal 向量，返回四元数 (x, y, z, w)"""
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, z_axis):
        return np.array([0, 0, 0, 1])
    if np.allclose(normal, -z_axis):
        return np.array([1, 0, 0, 0])
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R_mat = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    rot = R.from_matrix(R_mat)
    q = rot.as_quat()  # (x, y, z, w)
    return q

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

# 收集顶点数据，并剔除包含 NaN 的数据
vertices = []
for value in voxel_map.values():
    center = value.plane_ptr_.center.clone().cpu().numpy()
    radius = value.plane_ptr_.radius
    # print(radius)
    normal = value.plane_ptr_.normal.clone().cpu().numpy()
    normal = np.asarray(normal).reshape(-1)
    # 归一化法向量
    norm = np.linalg.norm(normal)
    if norm > 0:  # 避免除以零
        normal = normal / norm
    else:
        continue  # 如果法向量无效，跳过

    quat = normal_to_quaternion(normal)

    # 颜色 (红色示例)
    f_dc = [1.0, 0.0, 0.0]
    f_rest = [0.0] * 45  # 高阶 SH 填充 0
    opacity = 1.0
    scales = [-1.5*radius, -1.5*radius, -10]  # 薄片

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
