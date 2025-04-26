# utils/__init__.py
from typing import Final

# 常量
DIM_STATE: Final = 18  # 状态维度：3(旋转) + 3(位置) + 3(速度) + 3(陀螺仪偏差) + 3(加速度计偏差) + 3(重力)
INIT_COV: Final = 0.0000001  # 初始协方差值