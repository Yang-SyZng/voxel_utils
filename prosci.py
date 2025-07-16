from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import chi2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# Model基类
class Model(ABC):
    @abstractmethod
    def fit(self, pts):
        ...

    @abstractmethod
    def error(self, data):
        ...

    @abstractmethod
    def predict(self, data):
        ...

    @staticmethod
    @abstractmethod
    def get_complexity():
        ...

# PROSAC算法（去掉quality排序）
def prosac(data, model_type, tolerance, beta, eta0, psi,
           max_outlier_proportion, p_good_sample, max_number_of_draws,
           enable_n_star_optimization=True):
    num_points = data.shape[0]
    num_points_to_sample = model_type.get_complexity()
    chi2_value = chi2.isf(2 * psi, 1)

    def niter_ransac(p, epsilon, s, n_max):
        if n_max == -1:
            n_max = np.iinfo(np.int32).max
        if epsilon <= 0:
            return 1
        logarg = - np.exp(s * np.log(1 - epsilon))
        logval = np.log(1 + logarg)
        n = np.log(1 - p) / logval
        if logval < 0 and n < n_max:
            return np.ceil(n)
        return n_max

    def i_min(m, n, beta):
        mu = n * beta
        sigma = np.sqrt(n * beta * (1 - beta))
        return np.ceil(m + mu + sigma * np.sqrt(chi2_value))

    N = num_points
    m = num_points_to_sample
    T_N = niter_ransac(p_good_sample, max_outlier_proportion, num_points_to_sample, -1)
    I_N_min = (1 - max_outlier_proportion) * N

    n_star = N
    I_n_star = 0
    I_N_best = 0
    t = 0
    n = m
    T_n = T_N

    for i in range(m):
        T_n = T_n * (n - i) / (N - i)

    T_n_prime = 1
    k_n_star = T_N

    while ((I_N_best < I_N_min) or t <= k_n_star) and t < T_N and t <= max_number_of_draws:
        t = t + 1

        if (t > T_n_prime) and (n < n_star):
            T_nplus1 = (T_n * (n + 1)) / (n + 1 - m)
            n = n + 1
            T_n_prime = T_n_prime + np.ceil(T_nplus1 - T_n)
            T_n = T_nplus1

        # 随机选择样本点（去掉quality排序后的逻辑）
        pts_idx = np.random.choice(n, m, replace=False) if t > T_n_prime else np.append(np.random.choice(n - 1, m - 1, replace=False), n)

        sample = data[pts_idx]

        model = model_type()
        model.fit(sample)

        error = model.error(data)
        is_inlier = (error < tolerance)
        I_N = is_inlier.sum()

        if I_N > I_N_best:
            I_N_best = I_N
            n_best = N
            I_n_best = I_N
            best_model = model

            if enable_n_star_optimization:
                epsilon_n_best = I_n_best / n_best
                I_n_test = I_N
                for n_test in range(N, m, -1):
                    if not (n_test >= I_n_test):
                        raise RuntimeError('循环不变量违反：n_test >= I_n_test')
                    if ((I_n_test * n_best > I_n_best * n_test) and (I_n_test > epsilon_n_best * n_test + np.sqrt(
                            n_test * epsilon_n_best * (1 - epsilon_n_best) * chi2_value))):
                        if I_n_test < i_min(m, n_test, beta):
                            break
                        n_best = n_test
                        I_n_best = I_n_test
                        epsilon_n_best = I_n_best / n_best
                    I_n_test = I_n_test - is_inlier[n_test - 1]

            if I_n_best * n_star > I_n_star * n_best:
                if not (n_best >= I_n_best):
                    raise RuntimeError('断言不满足：n_best >= I_n_best')
                n_star = n_best
                I_n_star = I_n_best
                k_n_star = niter_ransac(1 - eta0, 1 - I_n_star / n_star, m, T_N)

    return best_model

# 平面模型
class PlaneModel(Model):
    def __init__(self):
        self.coefficients = None  # 平面方程：ax + by + cz + d = 0

    def fit(self, pts):
        """使用SVD拟合平面"""
        assert pts.shape[0] >= 3, "至少需要3个点来拟合平面"
        # 中心化点云
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        # SVD分解
        _, _, Vh = np.linalg.svd(centered_pts)
        # 法向量为最小奇异值对应的向量
        normal = Vh[-1]
        # 平面方程：ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, centroid)
        self.coefficients = [a, b, c, d]
        # 确保法向量朝向一致
        if c < 0:  # 使c为正，规范化输出
            self.coefficients = [-a, -b, -c, -d]

    def predict(self, data):
        """预测点在平面上的投影点"""
        a, b, c, d = self.coefficients
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        t = -(a * x + b * y + c * z + d) / (a**2 + b**2 + c**2)
        return np.column_stack([x + t * a, y + t * b, z + t * c])

    def error(self, data):
        """计算点到平面的距离"""
        a, b, c, d = self.coefficients
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        return np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

    @staticmethod
    def get_complexity():
        return 3  # 平面拟合需要3个点

# 保存点云到PCD文件
def save_point_cloud(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

# 可视化点云
def visualize_point_cloud(inliers, outliers, coefficients):
    # Matplotlib 3D散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='b', label='内点', s=10)
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='r', label='外点', s=10)
    
    # 绘制平面
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)
    a, b, c, d = coefficients
    Z = -(a * X + b * Y + d) / c
    ax.plot_surface(X, Y, Z, color='g', alpha=0.3)
    
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('使用PROSAC提取的平面')
    ax.legend()
    plt.savefig('plane_extraction_result.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Open3D可视化
    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inliers)
    inlier_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    outlier_pcd = o3d.geometry.PointCloud()
    outlier_pcd.points = o3d.utility.Vector3dVector(outliers)
    outlier_pcd.paint_uniform_color([1, 0, 0])  # 红色
    o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])

# 主函数
def main(data):
    # PROSAC参数
    tolerance = 0.1  # 点到平面的最大距离
    beta = 0.01  # 错误内点概率
    eta0 = 0.05
    psi = 0.05
    max_outlier_proportion = 0.5
    p_good_sample = 0.99
    max_number_of_draws = 10000

    # 运行PROSAC
    model = prosac(data, PlaneModel, tolerance, beta, eta0, psi,
                   max_outlier_proportion, p_good_sample, max_number_of_draws)

    # 提取内点和外点
    errors = model.error(data)
    inliers = data[errors < tolerance]
    outliers = data[errors >= tolerance]

    return model.coefficients

# 示例运行（假设data已提供）
# data = np.array(...) # 请替换为实际的data数据
# main(data)
if __name__ == '__main__':
    data = o3d.io.read_point_cloud("./points3D.ply")
    main(np.asarray(data.points))