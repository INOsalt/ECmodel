import pandas as pd
from scipy.optimize import curve_fit
import numpy as np

# 读取数据
data = pd.read_csv("RC.csv")
time_step = 0.5 * 3600  # 0.5小时，单位转换为秒
T_zone = data["TAIR_Zone1"].values
T_out = data["Tout"].values
T_wall = data["TSI_S11"].values

# 确保长度匹配
assert len(T_wall) == len(T_out) == len(T_zone), "输入数据长度不一致！"

# 定义拟合函数
def model(params, T_wall_prev, T_out, T_zone, dt):
    R_ext_wall, R_zone_wall, C = params
    Q_ext_wall = (T_out - T_wall_prev) / R_ext_wall
    Q_zone_wall = (T_wall_prev - T_zone) / R_zone_wall
    return T_wall_prev + dt / C * (Q_ext_wall - Q_zone_wall)


# 误差函数
def error_func(params, xdata, ydata, dt):
    # 解包 xdata
    T_wall, T_out, T_zone = xdata
    T_wall_prev = T_wall[:-1]
    T_wall_next = T_wall[1:]
    # 预测值
    predicted = model(params, T_wall_prev, T_out[1:], T_zone[1:], dt)
    return predicted - ydata  # ydata 是实际的 T_wall[1:]


# 初始猜测值
initial_guess = [5, 5, 1000]  # 假设热阻为5，热容为1000
bounds = (0, np.inf)  # 确保物理量为正

# 拟合
result = curve_fit(
    lambda xdata, R_ext_wall, R_zone_wall, C: error_func(
        [R_ext_wall, R_zone_wall, C], xdata, T_wall[1:], time_step
    ),
    xdata=(T_wall, T_out, T_zone),  # 输入数据作为元组
    ydata=T_wall[1:],  # 墙体温度的实际值
    p0=initial_guess,
    bounds=bounds
)

# 输出结果
R_ext_wall, R_zone_wall, C = result[0]
print("R_ext_wall:", R_ext_wall, "R_zone_wall:", R_zone_wall, "C:", C)

# 模型预测
T_wall_pred = model([R_ext_wall, R_zone_wall, C], T_wall[:-1], T_out[1:], T_zone[1:], time_step)

# 可视化
import matplotlib.pyplot as plt
plt.plot(T_wall[1:], label="Measured")
plt.plot(T_wall_pred, label="Predicted",alpha = 0.6)
plt.legend()
plt.show()
