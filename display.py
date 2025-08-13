# import open3d as o3d

# # 读取点云
# pcd = o3d.io.read_point_cloud("3.ply")

# # 创建八叉树
# octree = o3d.geometry.Octree(max_depth=1)
# octree.convert_from_point_cloud(pcd, size_expand=1)  # size_expand 控制包围盒外扩
# # 显示八叉树 + 原点云
# o3d.visualization.draw_geometries([octree])


import matplotlib.pyplot as plt
import numpy as np

def plot_ellipsoid(ax, center, radii, color, resolution=30):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color)  # 去掉alpha参数，默认不透明

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ticks = np.arange(0, 11, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.grid(True)
ax.set_axis_off()

num_ellipsoids = 20
np.random.seed(42)
for _ in range(num_ellipsoids):
    center = np.random.uniform(1, 9, size=3)
    radii = np.random.uniform(0.5, 2.0, size=3)
    color = np.random.rand(3)
    plot_ellipsoid(ax, center, radii, color)

plt.show()


