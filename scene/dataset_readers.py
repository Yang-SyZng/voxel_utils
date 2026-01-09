import numpy as np

from utils.graphics_utils import BasicPointCloud
import open3d as o3d

def fetchPly(path):
    pcd = o3d.io.read_point_cloud(path)
    positions = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.zeros_like(positions)

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
    else:
        normals = np.zeros_like(positions)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)