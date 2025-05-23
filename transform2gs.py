from main import main, read_yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import torch

args = read_yaml("config/cloud2voxel_mapping.yaml")
voxel_map = main(args)

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
for _, value in voxel_map.items():
    if value.octo_state_ == 0:
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
        rotation = np.asarray([value.plane_ptr_.x_normal.clone().cpu().numpy(), value.plane_ptr_.y_normal.clone().cpu().numpy(), value.plane_ptr_.normal.clone().cpu().numpy()]).T
        r = R.from_matrix(rotation)  # 从旋转矩阵创建旋转对象
        quat = r.as_quat()  # 转换为四元数 (x, y, z, w)
        quat_norm = quat * np.linalg.norm(quat, ord=2)
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
    else:
        for leaf_value in value.leaves_:
            if leaf_value is not None:
                if leaf_value.octo_state_ == 0:
                    center = leaf_value.plane_ptr_.center.clone().cpu().numpy()
                    radius = leaf_value.plane_ptr_.radius
                    # print(radius)
                    normal = leaf_value.plane_ptr_.normal.clone().cpu().numpy()
                    normal = np.asarray(normal).reshape(-1)
                    # 归一化法向量
                    norm = np.linalg.norm(normal)
                    if norm > 0:  # 避免除以零
                        normal = normal / norm
                    else:
                        continue  # 如果法向量无效，跳过
                    rotation = np.asarray([leaf_value.plane_ptr_.x_normal.clone().cpu().numpy(), leaf_value.plane_ptr_.y_normal.clone().cpu().numpy(), leaf_value.plane_ptr_.normal.clone().cpu().numpy()]).T
                    r = R.from_matrix(rotation)  # 从旋转矩阵创建旋转对象
                    quat = r.as_quat()  # 转换为四元数 (x, y, z, w)
                    quat_norm = quat * np.linalg.norm(quat, ord=2)
                    # 颜色 (红色示例)
                    f_dc = [1.0, 0.0, 0.0]
                    f_rest = [0.0] * 45  # 高阶 SH 填充 0
                    opacity = 1.0
                    # scales = [-5.*radius, -5.*radius, -10]  # 薄片
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
                else:
                    for leaf_leaf_value in leaf_value.leaves_:
                        if leaf_leaf_value is not None:
                            center = leaf_leaf_value.plane_ptr_.center.clone().cpu().numpy()
                            radius = leaf_leaf_value.plane_ptr_.radius
                            # print(radius)
                            normal = leaf_leaf_value.plane_ptr_.normal.clone().cpu().numpy()
                            normal = np.asarray(normal).reshape(-1)
                            # 归一化法向量
                            norm = np.linalg.norm(normal)
                            if norm > 0:  # 避免除以零
                                normal = normal / norm
                            else:
                                continue  # 如果法向量无效，跳过
                            rotation = np.asarray([leaf_leaf_value.plane_ptr_.x_normal.clone().cpu().numpy(), leaf_leaf_value.plane_ptr_.y_normal.clone().cpu().numpy(), leaf_leaf_value.plane_ptr_.normal.clone().cpu().numpy()]).T
                            r = R.from_matrix(rotation)  # 从旋转矩阵创建旋转对象
                            quat = r.as_quat()  # 转换为四元数 (x, y, z, w)
                            quat_norm = quat * np.linalg.norm(quat, ord=2)
                            # 颜色 (红色示例)
                            f_dc = [1.0, 0.0, 0.0]
                            f_rest = [0.0] * 45  # 高阶 SH 填充 0
                            opacity = 1.0
                            scales = [np.log(radius), np.log(radius), -10]  # 薄片
                            # scales = [radius, radius, -10]  # 薄片
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
                    else:
                        continue
            else:
                continue

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