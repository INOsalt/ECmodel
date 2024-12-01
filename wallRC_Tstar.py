import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'RCB.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 提取墙壁温度列和室内外温度列
wall_temp_columns = ['TSI_S4', 'TSI_S6',#roof floor
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10',#window 
                     'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'wall

file_path2 = 'RC.csv'


# 读取数据
df2 = pd.read_csv(file_path2)
# 空间热负荷 (kJ/hr -> W)
# 注意: 1 kJ/hr = 0.2778 W
Q_heat = df2['QHEAT_Zone1'].values * 0.2778
Q_cool = df2['QCOOL_Zone1'].values * 0.2778
Q_space = Q_heat - Q_cool
Q_in = df2['Qin_kJph'].values * 0.2778
# 通风供气温度 (°C)
vent_temp = df2['TAIR_fresh'].values
# 通风流量 (kg/hr -> kg/s)
# 注意: 1 kg/hr = 1/3600 kg/s
vent_flow = 520 / 3600
c_air = 1005  # 空气比热容 (J/kg·K)


# 时间步长
dt = 1800  # 0.5小时 -> 秒
def star_model(t, Rstar_win,Rstar_wall, Rair, Cstar,T_air):
    T_star_simulated = [T_air[0]]  # 初始化模拟温度
    for i in range(1, len(t)):
        T_air_t = T_air[i - 1]
        T_star_t = T_star_simulated[-1]

        Q_wall_star = 0
        for wall in wall_temp_columns:
            T_wall_in = data[wall].values
            T_wall_t = T_wall_in[i-1]
            if wall in ['TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10']:
                R_star = Rstar_win
            else:
                R_star = Rstar_wall
            Q_wall_star_temp = (T_wall_t - T_star_t) / R_star
            Q_wall_star = Q_wall_star + Q_wall_star_temp
        dT_star = dt/Cstar * (Q_wall_star - (T_star_t - T_air_t) / Rair)
        T_star_simulated.append(T_star_t + dT_star)
    return np.array(T_star_simulated)

#Rair Cair
T_air_measured = data['Tin'].values
T_star_measured = data['Tstar'].values
T_wall_ext = data['Tout'].values
t = np.arange(len(T_wall_ext))  # 时间步数
# 初始参数猜测
initial_guess = [0.001,0.001,0.001,1e7]
bounds = ([0.0001,0.0001,0.0002,10], [0.005,0.005,1,1e11])  # 参数范围

# # 拟合曲线
# popt, _ = curve_fit(lambda t, Rstar_win, Rstar_wall, Rair, Cstar:
#                     star_model(t, Rstar_win, Rstar_wall, Rair, Cstar, T_air_measured),
#                     t, T_star_measured , p0=initial_guess, bounds=bounds)
# Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt = popt
# print(Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt)
# Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt =[0.004999999999999999, 0.001226805281748122, 0.00020000000000000004, 19059395.842857108]
# T_star_simulated_cal = star_model(t,Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt, T_air_measured)

# # 绘制模拟与实际温度对比
# plt.figure(figsize=(8, 4))
# plt.plot(T_star_measured, label='Measured', linestyle='--', alpha=0.7)
# plt.plot(T_star_simulated_cal, label='Simulated', linestyle='-', alpha=0.7)
# plt.xlabel('Time Steps (0.5h each)')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.show()
#
# T_star_measured = data['Tstar'].values
# q_wall_zone = (T_star_simulated_cal - T_air_measured) / Rair_opt * 3600 / 1000
# q_wall_zone_measure = (T_star_measured - T_air_measured) / Rair_opt * 3600/1000
# # 提取 CSV 文件中的 QSURF
# q_surf = data['QSURF']
#
# # 绘制 Q_wall-zone 和 QSURF 比较图
# plt.figure(figsize=(12, 6))
# plt.plot(q_wall_zone, label='Q_wall-zone (Simulated)', linestyle='-', alpha=0.7)#, marker='o'
# plt.plot(q_wall_zone_measure, label='Q_wall-zone (Measured)', linestyle='-', alpha=0.7)#, marker='s'
# plt.plot(q_surf, label='Q_SURF (Measured)', linestyle='--', alpha=0.7)
# plt.xlabel('Time Steps (0.5h each)')
# plt.ylabel('Heat Flux (kJ/h)')
# plt.title('Comparison of Q_wall-zone and Q_SURF')
# plt.legend()
# plt.show()

# 定义 RC 模型函数（用于 curve_fit）
def rc_model(t, Rex, C, Rin, T_star_simulated, T_wall_ext, T_wall_ini,wall):
    T_wall_int_simulated = [T_wall_ini]  # 初始化模拟温度
    for i in range(1, len(t)):
        T_wall_ext_t = T_wall_ext[i-1]
        T_star_t = T_star_simulated[i-1]
        T_wall_t = T_wall_int_simulated[-1]
        if wall in ['TSI_S4', 'TSI_S6']:
            T_wall_ext_t = T_star_t
        dT_wall = dt / C * ((T_wall_ext_t - T_wall_t) / Rex - (T_wall_t - T_star_t) / Rin)
        T_wall_int_simulated.append(T_wall_t + dT_wall)
    return np.array(T_wall_int_simulated)

# 初始化结果存储
rc_params = {}
q_wall_zone_all = pd.DataFrame(index=data.index)
q_wall_zone_measure_all = pd.DataFrame(index=data.index)
Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt =[0.004999999999999999, 0.001226805281748122, 0.00020000000000000004, 19059395.842857108]

t = np.arange(len(data['TIME'].values))  # 时间步数
x_data = np.linspace(0, 4, 50)
T_wall_ext = data['Tout'].values
T_air = data['Tin'].values

# 针对每面墙
for wall in wall_temp_columns:
    T_wall_int_measured = data[wall].values
    t = np.arange(len(T_wall_ext))  # 时间步数
    if wall in ['TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10']:
        Rin = Rstar_opt_win
    else:
        Rin = Rstar_opt_wall

    # 初始参数猜测
    initial_guess = [0.001, 1e7]  # Rex, Rin, C
    bounds = ([0.0001, 1000], [0.05, 1e11])  # 参数范围

    # 拟合曲线
    popt, _ = curve_fit(lambda t, Rex, C:
                        rc_model(t, Rex, C, Rin, T_star_measured, T_wall_ext, T_wall_int_measured[0],wall),
                        t, T_wall_int_measured, p0=initial_guess, bounds=bounds)
    Rex_opt, C_opt = popt
    rc_params[wall] = (Rex_opt,Rin, C_opt)

    # 计算每个时间步长的 Q_wall-zone (单位转换为 kJ/h)
    T_wall_int_simulated = rc_model(t, Rex_opt, C_opt,Rin,T_star_measured, T_wall_ext, T_wall_int_measured[0],wall)

    # 绘制模拟与实际温度对比
    plt.figure(figsize=(8, 4))
    plt.plot(T_wall_int_measured, label='Measured', linestyle='--', alpha=0.7)
    plt.plot(T_wall_int_simulated, label='Simulated', linestyle='-', alpha=0.7)
    plt.xlabel('Time Steps (0.5h each)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Comparison for Wall {wall}')
    plt.legend()
    plt.show()



# 保存 RC 参数和比较结果
rc_params_df = pd.DataFrame(rc_params, index=['Rex', 'Rin', 'C']).T
rc_params_df.to_csv('rc_params_curvefit.csv', index=True)
# comparison_df = pd.DataFrame({'Q_wall-zone (kJ/h)': q_wall_zone_sum, 'Q_SURF (kJ/h)': q_surf})
# comparison_df.to_csv('comparison_curvefit.csv', index=False)
# 将 q_wall_zone_all 数据输出到 CSV 文件
q_wall_zone_all.to_csv('q_wall_zone_all.csv', index=True, header=True)
print("q_wall_zone_all 已保存到 q_wall_zone_all.csv")

print("RC 参数和比较结果已保存。")
