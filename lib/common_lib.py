import torch
from typing import List
from utils import DOUBLE, DEVICE, FLOAT64
from lib import DIM_STATE, INIT_COV
import time
import threading
#  VV \        VV \   AAAAAAAA\    LL\          UU\     UU\   EEEEEEEEEEE\  SSSSSSSS\
#   VV \      VV /   AA  ____AA\   LL |         UU |    UU |  EE  ______|  SS  ______|
#    VV \    VV /    AA /    AA |  LL |         UU |    UU |  EE |         SS /
#     VV \  VV /     AAAAAAAAAA |  LL |         UU |    UU |  EEEEEEEEEE\    SSSSSSS \
#      VV \VV /      AA  ____AA |  LL |         UU |    UU |  EE  ______|           SS \
#       VVVV /       AA |    AA |  LL |         UU |    UU |  EE |                  SS |
#        VV /        AA |    AA |  LLLLLLLLLL\   UUUUUUUU /   EEEEEEEEEEE\   SSSSSSSS /
#        \_/         \__|    \__|  \_________|   \_______/    \__________|   \_______/
# Created by zty 2025/05/03
MAX_INI_COUNT = 200
Eye3d = torch.eye(3, dtype=DOUBLE, device=DEVICE)
Eye3f = torch.eye(3, dtype=FLOAT64, device=DEVICE)
Zero3d = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
Zero3f = torch.zeros((3, 1), dtype=FLOAT64, device=DEVICE)
Lidar_offset_to_IMU = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
G_m_s2 = 9.81  # 重力加速度
Lidar_offset_to_IMU = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
# Vector3d shape(3, 1)
# Matrix3d shape(3, 3)

#   CCCCCCCCC\    LL\            AAAAAAAA\      SSSSSSSS\     SSSSSSSS\     EEEEEEEEEEE\  SSSSSSSS\
#  CC ________|   LL |          AA  ____AA\    SS  ______|   SS  ______|    EE  ______|  SS  ______|
#  CC |           LL |          AA /    AA |   SS /          SS /           EE |         SS /
#  CC |           LL |          AAAAAAAAAA |     SSSSSSS \     SSSSSSS \    EEEEEEEEEE\    SSSSSSS \
#  CC |           LL |          AA  ____AA |           SS \           SS \  EE  ______|           SS \
#  CC |           LL |          AA |    AA |           SS |           SS |  EE |                  SS |
#   CCCCCCCCC\    LLLLLLLLLL\   AA |    AA |    SSSSSSSS /     SSSSSSSS /   EEEEEEEEEEE\   SSSSSSSS /
#   \_________|   \_________|   \__|    \__|    \_______/      \_______/    \__________|   \_______/
# Created by zty 2025/04/26

# PointCloudXYZI = PointCloudXYZINormal
# PointXYZI = PointCloudXYZI
# PointCloudXYZI [x, y, z, intensity]
class BasedPoint:
    def __init__(self, points=None):
        if points is not None:
            if not isinstance(points, torch.Tensor):
                raise TypeError("points must be a torch.Tensor")
            if points.shape[1] != 3:
                raise ValueError("Each point must have 3 coordinates (x, y, z)")
            self.points = points
        else:
            self.points = torch.zeros((0, 3), dtype=DOUBLE, device=DEVICE)
            

    def add_points(self, points):
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a torch.Tensor")
        if points.shape[1] != 3:
            raise ValueError("Each point must have 3 coordinates (x, y, z)")
        self.points = torch.cat([self.points, points.to(dtype=DOUBLE, device=DEVICE)], dim=0)

    @property
    def size(self):
        return self.points.shape[0]

class PointXYZ(BasedPoint):
    def __init__(self, points=None):
        super().__init__(points=points)

class PointXYZI(PointXYZ):
    def __init__(self, points=None, intensity=None):
        if intensity is not None:
            if intensity.shape[1] != 1:
                raise ValueError("Each intensity must have 1 coordinates (intensity)")
            if points is None:
                raise TypeError(f"points is {points}")
            if not isinstance(intensity, torch.Tensor):
                raise TypeError("intensity must be a torch.Tensor")
        super().__init__(points=points)
        self.intensity = torch.zeros((self.points.shape[0], 1), dtype=DOUBLE, device=DEVICE) if intensity is None else intensity
    
    def add_points(self, points, intensity=None):
        super().add_points(points=points)
        if intensity is None:
            intensity = torch.zeros((points.shape[0], 1), dtype=DOUBLE, device=DEVICE)
        else:
            if not isinstance(intensity, torch.Tensor):
                raise TypeError("intensity must be a torch.Tensor")
        self.intensity = torch.cat([self.intensity, intensity.to(dtype=DOUBLE, device=DEVICE)], dim=0)

    def update_intensity(self, intensity):
        if not isinstance(intensity, torch.Tensor):
            raise TypeError("intensity must be a torch.Tensor")
        if intensity.shape[1] != 1:
            raise ValueError("Each intensity must have 1 coordinates (intensity)")
        if intensity.shape[0] != self.intensity.shape[0]:
            raise ValueError("Intensity must match shape")
        self.intensity = intensity

class PointXYZINormal(PointXYZI):
    def __init__(self, points=None, intensity=None, normals=None, curvature=None):
        if normals is not None:
            if normals.shape[1] != 3:
                raise ValueError("Each normals must have 3 coordinates (nx, ny, nz)")
            if not isinstance(normals, torch.Tensor):
                raise TypeError("normals must be a torch.Tensor")
        if curvature is not None:
            if curvature.shape[1] != 1:
                raise ValueError("Each curvature must have 1 coordinates (curvature)")
            if not isinstance(curvature, torch.Tensor):
                raise TypeError("curvature must be a torch.Tensor")
        if (normals is not None or curvature is not None) and points is None:
            raise TypeError(f"points is {points}")
        super().__init__(points=points, intensity=intensity)
        
        self.normals = torch.zeros((self.points.shape[0], 3), dtype=DOUBLE, device=DEVICE) if normals is None else normals
        self.curvature = torch.zeros((self.points.shape[0], 1), dtype=DOUBLE, device=DEVICE) if curvature is None else curvature

    def add_points(self, points, intensity=None, normals=None, curvature=None):
        super().add_points(points=points, intensity=intensity)
        if normals is None:
            normals = torch.zeros((points.shape[0], 3), dtype=DOUBLE, device=DEVICE)
        else:
            if not isinstance(normals, torch.Tensor):
                raise TypeError("normals must be a torch.Tensor")
        if curvature is None:
            curvature = torch.zeros((points.shape[0], 1), dtype=DOUBLE, device=DEVICE)
        else:
            if not isinstance(curvature, torch.Tensor):
                raise TypeError("curvature must be a torch.Tensor")
        self.normals = torch.cat([self.normals, normals.to(dtype=DOUBLE, device=DEVICE)], dim=0)
        self.curvature = torch.cat([self.curvature, curvature.to(dtype=DOUBLE, device=DEVICE)], dim=0)

    def update_normals(self, normals):
        if not isinstance(normals, torch.Tensor):
            raise TypeError("normals must be a torch.Tensor")
        if normals.shape[1] != 3:
            raise ValueError("Each normals must have 1 coordinates (nx, ny, nz)")
        if normals.shape[0] != self.normals.shape[0]:
            raise ValueError("normals must match shape")
        self.normals = normals

    def update_curvature(self, curvature):
        if not isinstance(curvature, torch.Tensor):
            raise TypeError("curvature must be a torch.Tensor")
        if curvature.shape[1] != 1:
            raise ValueError("Each curvature must have 1 coordinates (nx, ny, nz)")
        if curvature.shape[0] != self.curvature.shape[0]:
            raise ValueError("curvature must match shape")
        self.curvature = curvature
    
class pointWithCov(BasedPoint):
    def __init__(self, points=None, covs=None, point_world=None):
        if covs is not None:
            if covs.shape[1] != 3 and covs.shape[2] != 3:
                raise ValueError("Each covs shape must have 3x3")
            if not isinstance(covs, torch.Tensor):
                raise TypeError("covs must be a torch.Tensor")
        if point_world is not None:
            if point_world.shape[1] != 3:
                raise ValueError("Each point_world must have 3 coordinates (x, y, z)")
            if not isinstance(point_world, torch.Tensor):
                raise TypeError("point_world must be a torch.Tensor")
        if (covs is not None or point_world is not None) and points is None:
            raise TypeError(f"points is {points}") 
        if (points is not None and point_world is not None) and points.shape[0] != point_world.shape[0]:
            raise ValueError("point_world & points must match shape")
        super().__init__(points=points)
        self.covs = torch.zeros((self.points.shape[0], 3, 3), dtype=DOUBLE, device=DEVICE) if covs is None else covs
        self.point_world = torch.zeros((self.points.shape[0], 3), dtype=DOUBLE, device=DEVICE) if point_world is None else point_world

    def add_points(self, points, covs=None, point_world=None):
        super().add_points(points=points)
        if covs is None:
            covs = torch.zeros((points.shape[0], 3, 3), dtype=DOUBLE, device=DEVICE)
        else:
            if not isinstance(covs, torch.Tensor):
                raise TypeError("covs must be a torch.Tensor")
        if point_world is None:
            point_world = torch.zeros((points.shape[0], 3), dtype=DOUBLE, device=DEVICE)
        else:
            if not isinstance(point_world, torch.Tensor):
                raise TypeError("point_world must be a torch.Tensor")
        self.covs = torch.cat([self.covs, covs.to(dtype=DOUBLE, device=DEVICE)], dim=0)
        self.point_world = torch.cat([self.point_world, point_world.to(dtype=DOUBLE, device=DEVICE)], dim=0)

    def update_point_world(self, point_world):
        if not isinstance(point_world, torch.Tensor):
            raise TypeError("curvature must be a torch.Tensor")
        if point_world.shape[1] != 3:
            raise ValueError("Each point_world must have 3 coordinates (nx, ny, nz)")
        if point_world.shape[0] != self.point_world.shape[0]:
            raise ValueError("point_world must match shape")
        self.point_world = point_world
    
    def update_covs(self, covs):
        if not isinstance(covs, torch.Tensor):
            raise TypeError("curvature must be a torch.Tensor")
        if covs.shape[1] != 3 or covs.shape[2] != 3:
            raise ValueError("Each covs shape must have 3x3")
        if covs.shape[0] != self.covs.shape[0]:
            raise ValueError("curvature must match shape")
        self.covs = covs

class MeasureGroup:
    def __init__(self):
        self.lidar_beg_time = 0.0
        self.lidar = PointXYZINormal()
        self.imu = []
        
class StatesGroup:
    def __init__(self):
        """
        Initialize the state group.
        """

        self.rot_end = Eye3d.clone()  # 形状 (3, 3)
        self.pos_end = Zero3d.clone()
        self.vel_end = Zero3d.clone()
        self.bias_g = Zero3d.clone()
        self.bias_a = Zero3d.clone()
        self.gravity = Zero3d.clone()
        self.cov = torch.eye(DIM_STATE, dtype=DOUBLE, device=DEVICE) * INIT_COV  # 形状 (18, 18)
        
    def __add__(self, state_add: torch.Tensor):
        """
        Add a state increment to the current state.

        Args:
            state_add (torch.Tensor): Shape (DIM_STATE, 1), state increment.

        Returns:
            StatesGroup: New state group with updated values.
        """
        new_state = StatesGroup()
        new_state.rot_end = self.rot_end * Exp(state_add[0, 0], state_add[1, 0], state_add[2, 0])
        new_state.pos_end = self.pos_end + state_add[3: 6].reshape(3, 1)
        new_state.vel_end = self.vel_end + state_add[6: 9].reshape(3, 1)
        new_state.bias_g = self.bias_g + state_add[9: 12].reshape(3, 1)
        new_state.bias_a = self.bias_a + state_add[12: 15].reshape(3, 1)
        new_state.gravity = self.gravity + state_add[15: 18].reshape(3, 1)
        new_state.cov = self.cov
        return new_state

    def __iadd__(self, state_add):
        """
        In-place addition of a state increment.

        Args:
            state_add (torch.Tensor): Shape (DIM_STATE, 1), state increment.

        Returns:
            StatesGroup: Self with updated values.
        """
        self.rot_end = self.rot_end * Exp(state_add[0, 0], state_add[1, 0], state_add[2, 0])
        self.pos_end += state_add[3:6].reshape(3, 1)
        self.vel_end += state_add[6:9].reshape(3, 1)
        self.bias_g += state_add[9:12].reshape(3, 1)
        self.bias_a += state_add[12:15-DIM_STATE].reshape(3, 1)
        self.gravity += state_add[15:18].reshape(3, 1)

        return self

    def __sub__(self, other: 'StatesGroup'):
        """
        Compute the difference between two states.

        Args:
            other (StatesGroup): Another state group.

        Returns:
            torch.Tensor: Shape (DIM_STATE, 1), state difference.
        """
        diff = torch.zeros(DIM_STATE, 1, dtype=DOUBLE, device=DEVICE)

        rotd = other.rot_end.T * self.rot_end
        diff[0:3] = Log(rotd)
        diff[3:6] = self.pos_end - other.pos_end
        diff[6:9] = self.vel_end - other.vel_end
        diff[9:12] = self.bias_g - other.bias_g
        diff[12:15] = self.bias_a - other.bias_a
        diff[15:18] = self.gravity - other.gravity

        return diff

    def resetpose(self):
        """
        Reset pose-related states to zero.
        """
        self.rot_end = torch.eye(3, dtype=DOUBLE, device=DEVICE)
        self.pos_end = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
        self.vel_end = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)

class ImuProcess:
    def __init__(self):
        self.b_first_frame_: bool = True
        self.imu_need_init_: bool = True
        self.start_timestamp_: float = -1
        self.imu_en: bool = True
        self.init_iter_num: int = 1
        self.cov_acc = torch.tensor([[0.1], [0.1], [0.1]], dtype=DOUBLE, device=DEVICE)
        self.cov_gyr = torch.tensor([[0.1], [0.1], [0.1]], dtype=DOUBLE, device=DEVICE)

        self.cov_acc_scale = torch.tensor([[1], [1], [1]], dtype=DOUBLE, device=DEVICE)
        self.cov_gyr_scale = torch.tensor([[1], [1], [1]], dtype=DOUBLE, device=DEVICE)
        
        self.cov_bias_gyr = torch.tensor([[0.1], [0.1], [0.1]], dtype=DOUBLE, device=DEVICE)
        self.cov_bias_acc = torch.tensor([[0.1], [0.1], [0.1]], dtype=DOUBLE, device=DEVICE)

        self.mean_acc = torch.tensor([[0], [0], [-1.0]], dtype=DOUBLE, device=DEVICE)
        self.mean_gyr = torch.tensor([[0], [0], [0]], dtype=DOUBLE, device=DEVICE)

        self.angvel_last = Zero3d.clone()
        self.Lid_offset_to_IMU = Lidar_offset_to_IMU.clone()
        self.Lid_rot_to_IMU = Eye3d.clone()
        self.first_lidar_time = None
        # self.last_imu_
        self.time_last_scan_: float = -1.0
    def Reset(self):
        self.mean_acc = torch.tensor([[0], [0], [-1.0]], dtype=DOUBLE, device=DEVICE)
        self.mean_gyr = torch.tensor([[0], [0], [0]], dtype=DOUBLE, device=DEVICE)
        self.angvel_last = Zero3d.clone()
        self.imu_need_init_: bool = True
        self.start_timestamp_: float = -1
        self.init_iter_num: int = 1
        # v_imu_.clear();
        # IMUpose.clear();  
        # last_imu_.reset(new sensor_msgs::Imu());
        # cur_pcl_un_.reset(new PointCloudXYZI());
    
    def IMU_init(self, meas: MeasureGroup, state_inout: StatesGroup, N: int):
        """
        初始化 IMU 数据，包括重力和协方差。

        Args:
            meas (MeasureGroup): 测量数据。
            state_inout (StatesGroup): 状态对象。
            N (int): IMU 数据计数器。
        """

        cur_acc = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)
        cur_gyr = torch.zeros((3, 1), dtype=DOUBLE, device=DEVICE)

        if self.b_first_frame_:
            self.Reset()
            N = 1
            self.b_first_frame_ = False
            imu = meas.imu[0]  # 取第一个 IMU 数据
            self.mean_acc = imu.linear_acceleration
            self.mean_gyr = imu.angular_velocity
            self.first_lidar_time = meas.lidar_beg_time

        for imu in meas.imu:
            cur_acc = imu.linear_acceleration
            cur_gyr = imu.angular_velocity

            # 更新均值
            self.mean_acc += (cur_acc - self.mean_acc) / N
            self.mean_gyr += (cur_gyr - self.mean_gyr) / N

            # 更新协方差
            diff_acc = cur_acc - self.mean_acc
            diff_gyr = cur_gyr - self.mean_gyr
            self.cov_acc = self.cov_acc * (N - 1.0) / N + diff_acc * diff_acc.t() * (N - 1.0) / (N * N)
            self.cov_gyr = self.cov_gyr * (N - 1.0) / N + diff_gyr * diff_gyr.t() * (N - 1.0) / (N * N)

            # print(f"acc norm: {torch.norm(cur_acc).item()} {torch.norm(self.mean_acc).item()}")

            N += 1

        state_inout.gravity = -self.mean_acc / torch.norm(self.mean_acc) * G_m_s2

        # 设置初始旋转和偏差
        state_inout.rot_end = Eye3d.clone()
        state_inout.bias_g = self.mean_gyr

        self.last_imu_ = meas.imu[-1]

        return state_inout, N
    
    def set_extrinsic(self, T: torch.Tensor = None, 
                    transl: torch.Tensor = None, 
                    rot: torch.Tensor = None):
        """
        set IMU extrinsic
        
        Args:
            T (torch.Tensor): 4x4Transformation Matrix
            transl (torch.Tensor): 3D translation vector
            rot (torch.Tensor): 3x3 rotation matrix
        """
        if T is not None:
            assert T.shape == (4, 4), "T must be 4x4 matrix"
            self.Lid_offset_to_IMU = T[:3, 3]
            self.Lid_rot_to_IMU = T[:3, :3]
        
        elif transl is not None and rot is None:
            assert transl.shape == (3, 1), "transl must be 3D vector"
            self.Lid_offset_to_IMU = transl
            self.Lid_rot_to_IMU = Eye3d.clone()
        
        elif transl is not None and rot is not None:
            assert transl.shape == (3, 1), "transl must be 3D vector"
            assert rot.shape == (3,3), "rot must be 3x3 matrix"
            self.Lid_offset_to_IMU = transl
            self.Lid_rot_to_IMU = rot
        
        else:
            raise ValueError("Invalid arguments: must provide T or transl (+ optional rot)")
    
    def set_gyr_cov_scale(self, scaler: torch.Tensor):
        self.cov_gyr_scale = scaler

    def set_acc_cov_scale(self, scaler: torch.Tensor):
        self.cov_acc_scale = scaler
    
    def set_gyr_bias_cov(self, b_g: torch.Tensor):
        self.cov_bias_gyr = b_g

    def set_acc_bias_cov(self, b_a: torch.Tensor):
        self.cov_bias_acc = b_a

    def only_propag(self, meas: MeasureGroup, state_inout: StatesGroup):

        pcl_beg_time = meas.lidar_beg_time

        # 设置输出点云
        pcl_out = meas.lidar
        
        # if len(pcl_out) > 0:
        #     pcl_end_time = pcl_beg_time + pcl_out.points[-1, 3] / 1000.0
        # else:
        #     pcl_end_time = pcl_beg_time

        # 计算时间差 dt
        if self.b_first_frame_:
            dt = 0.1
            self.b_first_frame_ = False
            self.time_last_scan_ = pcl_beg_time
            # print("测试", pcl_beg_time)
        else:
            # dt = pcl_beg_time - self.time_last_scan_
            dt = 5.
            self.time_last_scan_ = pcl_beg_time
            # print("测试时间戳，dt", self.time_last_scan_, dt)

        Exp_f = Exp(state_inout.bias_g, dt)  # 使用新的 Exp 函数
        # print("Exp_f", Exp_f)
        F_x = torch.eye(DIM_STATE, dtype=DOUBLE, device=DEVICE)
        cov_w = torch.zeros((DIM_STATE, DIM_STATE), dtype=DOUBLE, device=DEVICE)

        # 设置 F_x 的子块
        F_x[0:3, 0:3] = Exp_f  # 旋转部分
        F_x[0:3, 9:12] = Eye3d.clone() * dt
        F_x[3:6, 6:9] = Eye3d.clone() * dt

        # 设置噪声协方差 cov_w
        for i in range(3):
            cov_w[9 + i, 9 + i] = (self.cov_gyr[i] * (dt ** 2)).item()
        for i in range(3):
            cov_w[6 + i, 6 + i] = (self.cov_acc[i] * (dt ** 2)).item()

        # 更新协方差
        state_inout.cov = F_x @ state_inout.cov @ F_x.T + cov_w
        # 更新状态
        state_inout.rot_end = state_inout.rot_end @ Exp_f
        state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt
        return state_inout, pcl_out
    
    def Process(self, meas: MeasureGroup, stat: StatesGroup):
        # if (meas.imu.empty() && imu_en) {
        #     return;
        # }
        # ROS_ASSERT(meas.lidar != nullptr);

        if self.imu_need_init_ and self.imu_en:
            stat, self.init_iter_num = self.IMU_init(meas, stat, self.init_iter_num)
            self.imu_need_init_ = True
            # self.last_imu_ = meas.imu

            if self.init_iter_num > MAX_INI_COUNT:
                self.cov_acc *= (G_m_s2 / self.mean_acc.norm()) ** 2
                self.imu_need_init_ = False
                self.cov_acc = Eye3d.clone() * self.cov_acc_scale
                self.cov_gyr = Eye3d.clone() * self.cov_gyr_scale

            return
        if self.imu_en:
            print("Use IMU")
            # UndistortPcl(meas, stat, *cur_pcl_un_);
            # last_imu_ = meas.imu.back();
        else:
            print("No IMU, use constant velocity model")
            self.cov_acc = self.cov_acc_scale.clone()
            self.cov_gyr = self.cov_gyr_scale.clone()
            return self.only_propag(meas=meas, state_inout=stat)
class PointCloud2Msg:
    def __init__(self, timestamp=None, data=None):
        # 如果未提供时间戳，则使用当前时间
        self.header = {'stamp': timestamp if timestamp is not None else time.time()}

    def toSec(self):
        # 获取时间戳
        return self.header['stamp']
class TimestampUpdater:
    def __init__(self, Duration: float):
        self.timestamp = PointCloud2Msg()  # 初始化时间戳
        self.running = True  # 启动暂停
        self.Duration = Duration
    def update_timestamp(self):
        while self.running:
            self.timestamp.header['stamp'] = time.time()  # 更新时间戳
            print(f"Updated Timestamp: {self.timestamp.toSec()}")
            time.sleep(self.Duration)

    def start(self):
        thread = threading.Thread(target=self.update_timestamp)
        thread.daemon = True  # 设置为守护线程，确保程序退出时它也退出
        thread.start()

    def stop(self):
        self.running = False
# FFFFFFFF\    UU\     UU\    NN\    NN\     CCCCCCCCC\    TTTTTTTTTT\   IIIIII\      OOOOOOOO\      NN\     NN\     SSSSSSSS\
# FF  _____|   UU |    UU |   NNN\   NN |   CC ________|       TT  __|     II  _|    OO _____OO \    NNN\    NN |   SS  ______|
# FF |         UU |    UU |   NN NN  NN |   CC |               TT |        II |     OO /      OO |   NNNN\   NN |   SS /
# FFFFF\       UU |    UU |   NN \N\ NN |   CC |               TT |        II |     OO |      OO |   NN NN\  NN |    SSSSSSS \
# FF  __|      UU |    UU |   NN |\NNNN |   CC |               TT |        II |     OO |      OO |   NN | NN\NN |           SS \
# FF |         UU |    UU |   NN | \NNN |   CC |               TT |        II |      OO \    OO /    NN |  NNNN |           SS |
# FF |          UUUUUUUU /    NN |  \NN |    CCCCCCCCC\        TT |      IIIIII\      OOOOOOOO /     NN |   NNN |    SSSSSSSS /
# \__|          \_______/     \__|   \__|    \_________|       \__|      \______|     \_______|      \__|   \___|    \_______/
# Created by zty 2025/04/26

def sync_packages(meas: MeasureGroup):
    pass

def Exp(*args) -> torch.Tensor:
    """
    计算 3x3 旋转矩阵，使用Roderigous Tranformation。

    Args:
        v1, v2, v3 (float): 3D 向量分量。

    Returns:
        torch.Tensor: 3x3 旋转矩阵。
    """
    Eye3 = torch.eye(3, dtype=DOUBLE, device=DEVICE)
    if len(args) == 1:
        # Exp(ang: Tensor)
        ang = args[0].reshape(-1, 1)
        norm = torch.norm(ang)
        if norm > 1e-7:
            r_ang = ang / norm
            K = torch.tensor([
                [0, -r_ang[2, 0], r_ang[1, 0]],
                [r_ang[2, 0], 0, -r_ang[0, 0]],
                [-r_ang[1, 0], r_ang[0, 0], 0]
            ], dtype=DOUBLE, device=DEVICE)
            return Eye3 + torch.sin(norm) * K + (1.0 - torch.cos(norm)) * K @ K
        else:
            return Eye3
    
    elif len(args) == 2:
        # Exp(ang_vel: Tensor, dt: float)
        ang_vel, dt = args
        ang_vel = ang_vel.reshape(-1, 1)
        norm = torch.norm(ang_vel)
        if norm > 1e-7:
            r_ang = ang_vel / norm
            K = torch.tensor([
                [0, -r_ang[2, 0], r_ang[1, 0]],
                [r_ang[2, 0], 0, -r_ang[0, 0]],
                [-r_ang[1, 0], r_ang[0, 0], 0]
            ], dtype=DOUBLE, device=DEVICE)
            r_angle = norm * dt
            return Eye3 + torch.sin(r_angle) * K + (1.0 - torch.cos(r_angle)) * K @ K
        else:
            return Eye3

    elif len(args) == 3:
        # Exp(v1, v2, v3)
        v1, v2, v3 = args
        v = torch.tensor([v1, v2, v3], dtype=DOUBLE, device=DEVICE).reshape(3, 1)
        norm = torch.norm(v)
        if norm > 1e-5:
            r_ang = v / norm
            K = torch.tensor([
                [0, -r_ang[2, 0], r_ang[1, 0]],
                [r_ang[2, 0], 0, -r_ang[0, 0]],
                [-r_ang[1, 0], r_ang[0, 0], 0]
            ], dtype=DOUBLE, device=DEVICE)
            return Eye3 + torch.sin(norm) * K + (1.0 - torch.cos(norm)) * K @ K
        else:
            return Eye3
    else:
        raise ValueError(f"Unsupported number of arguments: {len(args)}")

def Log(R: torch.Tensor) -> torch.Tensor:
    """
    Calculate the axis-angle vector of a 3x3 rotation matrix (Log Map).

    Args:
        R (torch.Tensor): 3x3 Rotation Matrix。

    Returns:
        torch.Tensor: 3x1 Axis angle vector.
    """
    # 计算迹
    trace_R = R.trace()

    # 计算旋转角度 theta
    theta = 0.0 if trace_R > 3.0 - 1e-6 else torch.acos(0.5 * (trace_R - 1))

    # 计算反对称矩阵的向量表示 K
    K = torch.tensor([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=DOUBLE, device=DEVICE).reshape(3, 1)

    # 根据 theta 大小选择返回结果
    if torch.abs(theta) < 0.001:
        return 0.5 * K
    else:
        return (0.5 * theta / torch.sin(theta)) * K