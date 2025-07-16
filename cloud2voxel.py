import argparse
from argparse import Namespace
import yaml
import torch
from typing import Final, List, Dict
import open3d as o3d
import numpy as np
import voxel_utils as vx
from voxel_utils import VOXEL_LOC, OctoTree
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

def cloud2voxel(args: Namespace, input_pcd=None):
    file_path = args.file_path
    file_format = args.file_format
    
    # mapping algorithm params
    layer_point_size = args.layer_point_size
    layer_size = layer_point_size
    max_layer = args.max_layer
    max_voxel_size = args.voxel_size
    filter_size_surf_min = args.down_sample_size
    min_eigen_value = args.plannar_threshold # min_eigen_value
    
    voxel_map: Dict[VOXEL_LOC, OctoTree] = {}
    if input_pcd is None:
        pcd = readPointCloud(file_path, file_format)
    else:
        pcd = input_pcd
    vx.buildVoxelMap(pcd, max_voxel_size, max_layer, min_eigen_value, voxel_map)
    return voxel_map
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
                        voxel_num += 1
                    else:
                        for leaf_leaf_value in leaf_value.leaves_:
                            if leaf_leaf_value is not None:
                                voxel_num += 1
                        else:
                            continue
                else:
                    continue
    print(voxel_num)