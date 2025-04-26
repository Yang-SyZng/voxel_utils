import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for point cloud processing and mapping')
    
    # Common parameters
    parser.add_argument('--lid_topic', type=str, default='/pointcloud',
                        help='LiDAR topic name')
    parser.add_argument('--imu_topic', type=str, default='/livox/imu',
                        help='IMU topic name')
    # Pointcloud parameters
    parser.add_argument('--file_path', type=str, 
                        default='/home/y/projections/catkin_ws_cloud2VoxelMap/points3D.ply',
                        help='Path to point cloud file')
    parser.add_argument('--file_format', type=str, default='ply',
                        choices=['pcd', 'ply'],
                        help='Point cloud file format')
    # Preprocess parameters
    parser.add_argument('--lidar_type', type=int, default=2,
                        choices=[1, 2, 3],
                        help='LiDAR type: 1 for Livox, 2 for Velodyne, 3 for L515')
    parser.add_argument('--scan_line', type=int, default=32,
                        help='Number of scan lines')
    parser.add_argument('--blind', type=int, default=0,
                        help='Blind parameter')
    parser.add_argument('--point_filter_num', type=int, default=1,
                        help='Point filter number')
    parser.add_argument('--calib_laser', type=bool, default=False,
                        help='Enable laser calibration for KITTI dataset')
    # Mapping parameters
    parser.add_argument('--down_sample_size', type=float, default=0.5,
                        help='Down sample size')
    parser.add_argument('--max_iteration', type=int, default=3,
                        help='Maximum iterations')
    parser.add_argument('--voxel_size', type=float, default=2.0,
                        help='Voxel size')
    parser.add_argument('--max_layer', type=int, default=2,
                        help='Maximum layer number')
    parser.add_argument('--layer_point_size', type=int, nargs=5, 
                        default=[5, 5, 5, 5, 5],
                        help='Layer point sizes')
    parser.add_argument('--plannar_threshold', type=float, default=0.01,
                        help='Planar threshold')
    parser.add_argument('--max_points_size', type=int, default=1000,
                        help='Maximum points size')
    parser.add_argument('--max_cov_points_size', type=int, default=1000,
                        help='Maximum covariance points size')
    # Noise model parameters
    parser.add_argument('--ranging_cov', type=float, default=0.04,
                        help='Ranging covariance')
    parser.add_argument('--angle_cov', type=float, default=0.1,
                        help='Angle covariance')
    parser.add_argument('--acc_cov_scale', type=float, default=1.0,
                        help='Accelerometer covariance scale')
    parser.add_argument('--gyr_cov_scale', type=float, default=0.5,
                        help='Gyroscope covariance scale')
    # IMU parameters
    parser.add_argument('--imu_en', type=bool, default=False,
                        help='Enable IMU')
    parser.add_argument('--extrinsic_T', type=float, nargs=3, 
                        default=[0.04165, 0.02326, -0.0284],
                        help='Extrinsic translation parameters')
    parser.add_argument('--extrinsic_R', type=float, nargs=9, 
                        default=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                        help='Extrinsic rotation matrix')
    # Visualization parameters
    parser.add_argument('--pub_voxel_map', type=bool, default=True,
                        help='Publish voxel map')
    parser.add_argument('--publish_max_voxel_layer', type=int, default=3,
                        help='Maximum voxel layer to publish')
    parser.add_argument('--pub_point_cloud', type=bool, default=False,
                        help='Publish point cloud')
    parser.add_argument('--dense_map_enable', type=bool, default=False,
                        help='Enable dense map')
    parser.add_argument('--pub_point_cloud_skip', type=int, default=1,
                        help='Point cloud publishing skip rate')
    # Result parameters
    parser.add_argument('--write_kitti_log', type=bool, default=False,
                        help='Write KITTI log')
    parser.add_argument('--result_path', type=str, 
                        default='/home/b/kitt_log.txt',
                        help='Path for result log')
    # # 检查是否有命令行参数
    # if len(sys.argv) == 1:  # 如果没有提供参数
    #     parser.print_help()  # 打印帮助信息
    args = parser.parse_args()