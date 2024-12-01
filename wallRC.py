import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("RC.csv")
time_step = 0.5 #* 3600  # 0.5小时，单位转换为秒
T_zone = data["TAIR_Zone1"].values
T_out = data["Tout"].values
T_wall = data["TSI_S11"].values

# 确保长度匹配
assert len(T_wall) == len(T_out) == len(T_zone), "输入数据长度不一致！"

# 定义模型函数
def model(T_wall_prev, T_out, T_zone, R_ext_wall, R_zone_wall, C, dt):
    Q_ext_wall = (T_out - T_wall_prev) / R_ext_wall
    Q_zone_wall = (T_wall_prev - T_zone) / R_zone_wall
    return T_wall_prev + dt / C * (Q_ext_wall - Q_zone_wall)

# 包装函数，供 curve_fit 使用
def model_for_curve_fit(xdata, R_ext_wall, R_zone_wall, C):
    T_wall_prev, T_out, T_zone = xdata
    dt = time_step
    return model(T_wall_prev, T_out, T_zone, R_ext_wall, R_zone_wall, C, dt)

# 准备数据
xdata = (T_wall[:-1], T_out[1:], T_zone[1:])  # 输入前一时刻的温度
ydata = T_wall[1:]  # 墙体下一时刻的温度

# 初始猜测值
initial_guess = [5, 5, 1000]  # 假设热阻为5，热容为1000
bounds = (0, np.inf)  # 确保物理量为正

# 拟合
result = curve_fit(model_for_curve_fit, xdata, ydata, p0=initial_guess, bounds=bounds)

# 输出结果
R_ext_wall, R_zone_wall, C = result[0]
print("R_ext_wall:", R_ext_wall, "R_zone_wall:", R_zone_wall, "C:", C)

# 模型预测
T_wall_pred = model_for_curve_fit(xdata, R_ext_wall, R_zone_wall, C)

# 可视化
plt.plot(ydata, label="Measured")
plt.plot(T_wall_pred, label="Predicted", alpha=0.6)
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Wall Temperature")
plt.title("Wall Temperature Prediction vs Measured")
plt.show()
