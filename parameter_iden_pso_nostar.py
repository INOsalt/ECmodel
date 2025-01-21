import pandas as pd
import numpy as np
from pyswarm import pso  # PSO optimization library
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'RCB.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 提取墙壁温度列和室内外温度列
wall_temp_columns = ['TSI_S4', 'TSI_S6',  # roof
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10',  # window
                     'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']  # ext wall
file_path2 = 'RC.csv'
df2 = pd.read_csv(file_path2)

# 空间热负荷 (kJ/hr -> W)
Q_heat = df2['QHEAT_Zone1'].values * 0.2778
Q_cool = df2['QCOOL_Zone1'].values * 0.2778
Q_space = Q_heat - Q_cool
Q_in = df2['Qin_kJph'].values * 0.2778

# 通风供气温度 (°C)
vent_temp = df2['TAIR_fresh'].values
vent_flow = 520 / 3600  # 通风流量 (kg/s)
c_air = 1005  # 空气比热容 (J/kg·K)

# 时间步长
dt = 1800  # 0.5小时 -> 秒

def air_model(t, Rstar_win, Rstar_wall, C_air, T_air_ini):
    T_air_simulated = [T_air_ini]
    for i in range(1, len(t)):
        T_air_t = T_air_simulated[- 1]
        Q_wall = 0
        for wall in wall_temp_columns:
            T_wall_in = data[wall].values
            T_wall_t = T_wall_in[i-1]
            if wall in ['TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10']:
                R_wall = Rstar_win
            else:
                R_wall = Rstar_wall
            Q_wall_star_temp = (T_wall_t - T_air_t) / R_wall
            Q_wall = Q_wall + Q_wall_star_temp

        # 空间负荷
        Q_space_t = Q_space[i - 1]  # 直接等于输入的 Q_heat
        # 通风系统热流 (Q_vent)
        temp_diff = vent_temp[i - 1] - T_air_t
        # temp_diff = np.clip(temp_diff, -50, 50)  # 限制温差范围在 -50 到 50 之间
        # try:
        Q_vent_t = vent_flow * c_air * temp_diff
        # except OverflowError:
        #     print(f"Overflow encountered at t={i}, setting Q_vent to 0.")
        #     Q_vent_t = 0
        Q_in_t = Q_in[i - 1]
        Q_air = Q_wall + Q_space_t + Q_vent_t + Q_in_t
        dT_air = dt / C_air * Q_air
        T_air_simulated.append(T_air_t + dT_air)

    return np.array(T_air_simulated)

# 目标函数
T_air_measured = data['Tin'].values
t = np.arange(len(T_air_measured))  # 时间步数

def fitness_function(params):
    Rstar_win, Rstar_wall, C_air = params
    T_air_simulated = air_model(t, Rstar_win, Rstar_wall, C_air, T_air_measured[0])
    mse = np.mean((T_air_measured - T_air_simulated) ** 2)  # 均方误差
    return mse

# 定义参数范围
lb = [1e-5, 1e-5, 10]  # 下界
ub = [0.1, 0.1, 1e10]  # 上界

# 使用PSO进行优化
best_params, best_score = pso(fitness_function, lb, ub, swarmsize=100, maxiter=500)

# 输出最佳参数
Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt, C_air_opt = best_params
print("Optimized Parameters:")
print(f"Rstar_win: {Rstar_opt_win}, Rstar_wall: {Rstar_opt_wall}, C_air: {C_air_opt}")
# 写入到文本文件
file_path = "pso_parameters.txt"  # 你可以根据需要修改文件路径
with open(file_path, "a") as f:  # "a" 模式确保文件不存在时会创建，且内容追加到文件末尾
    f.write("Optimized Parameters:\n")
    f.write(f"Rstar_win: {Rstar_opt_win}, Rstar_wall: {Rstar_opt_wall}, C_air: {C_air_opt}\n")

# 使用最佳参数计算模拟结果
T_air_simulated_cal = air_model(t, Rstar_opt_win, Rstar_opt_wall, C_air_opt, T_air_measured[0])

# 绘制模拟与实际温度对比
plt.figure(figsize=(8, 4))
plt.plot(T_air_measured, label='Measured', linestyle='--', alpha=0.7)
plt.plot(T_air_simulated_cal, label='Simulated (PSO)', linestyle='-', alpha=0.7)
plt.xlabel('Time Steps (0.5h each)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

Q_wall_zone = 0
Q_wall_measure = 0
for wall in wall_temp_columns:
    T_wall_in = data[wall].values
    T_wall_t = T_wall_in
    if wall in ['TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10']:
        R_wall = Rstar_opt_win
    else:
        R_wall = Rstar_opt_wall
    Q_wall_zone_temp = (T_wall_t - T_air_simulated_cal) / R_wall
    Q_wall_measure_temp = (T_wall_t - T_air_measured) / R_wall
    Q_wall_zone = Q_wall_zone + Q_wall_zone_temp
    Q_wall_measure = Q_wall_measure + Q_wall_measure_temp

# 提取 CSV 文件中的 QSURF
q_surf = data['QSURF']

# 绘制 Q_wall-zone 和 QSURF 比较图
plt.figure(figsize=(12, 6))
plt.plot(Q_wall_zone, label='Q_wall-zone (Simulated)', linestyle='-', alpha=0.7)#, marker='o'
plt.plot(Q_wall_measure, label='Q_wall-zone (Measured)', linestyle='-', alpha=0.7)#, marker='s'
plt.plot(q_surf, label='Q_SURF (Measured)', linestyle='--', alpha=0.7)
plt.xlabel('Time Steps (0.5h each)')
plt.ylabel('Heat Flux (kJ/h)')
plt.title('Comparison of Q_wall-zone and Q_SURF')
plt.legend()
plt.show()