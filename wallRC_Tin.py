import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'RCB.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 提取墙壁温度列和室内外温度列
wall_temp_columns = ['TSI_S4', 'TSI_S6',
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10', 'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'TSI_S1', 'TSI_S2', 'TSI_S3',  'TSI_S5',# ext wall


# 时间步长
dt = 1800  # 0.5小时 -> 秒

# 定义 RC 模型函数（用于 curve_fit）
def rc_model(t, Rex, Rin, C, T_wall_ext, T_air, T_init):
    T_wall_int_simulated = [T_init]  # 初始化模拟温度
    for i in range(1, len(t)):
        T_int_t = T_wall_int_simulated[-1]
        dT = dt / C * ((T_wall_ext[i-1] - T_int_t) / Rex - (T_int_t - T_air[i-1]) / Rin)
        T_wall_int_simulated.append(T_int_t + dT)
    return np.array(T_wall_int_simulated)

# 初始化结果存储
rc_params = {}
q_wall_zone_all = pd.DataFrame(index=data.index)
q_wall_zone_measure_all = pd.DataFrame(index=data.index)

# 针对每面墙拟合 RC 参数
for wall in wall_temp_columns:
    T_wall_int_measured = data[wall].values
    T_wall_ext = data['Tout'].values
    T_air = data['Tin'].values
    t = np.arange(len(T_wall_ext))  # 时间步数
    if wall in ['TSI_S4', 'TSI_S6',]:
        T_wall_ext = data['Tout'].values


    # 初始参数猜测
    initial_guess = [0.005, 0.005, 1000000]  # Rex, Rin, C
    bounds = ([0.001, 0.001, 1000], [0.03, 0.03, 6000000])  # 参数范围

    # 拟合曲线
    popt, _ = curve_fit(
        lambda t, Rex, Rin, C: rc_model(t, Rex, Rin, C, T_wall_ext, T_air, T_wall_int_measured[0]),
        t, T_wall_int_measured, p0=initial_guess, bounds=bounds
    )
    Rex_opt, Rin_opt, C_opt = popt
    rc_params[wall] = (Rex_opt, Rin_opt, C_opt)

    # 计算每个时间步长的 Q_wall-zone (单位转换为 kJ/h)
    T_wall_int_simulated = rc_model(t, Rex_opt, Rin_opt, C_opt, T_wall_ext, T_air, T_wall_int_measured[0])
    q_wall_zone = (T_wall_int_simulated - T_air) / Rin_opt * 3600/1000
    q_wall_zone_all[wall] = q_wall_zone
    q_wall_zone_measure = (T_wall_int_measured - T_air) / Rin_opt * 3600/1000
    q_wall_zone_measure_all[wall] = q_wall_zone_measure

    # 绘制模拟与实际温度对比
    plt.figure(figsize=(8, 4))
    plt.plot(T_wall_int_measured, label='Measured', linestyle='--', alpha=0.7)
    plt.plot(T_wall_int_simulated, label='Simulated', linestyle='-', alpha=0.7)
    plt.xlabel('Time Steps (0.5h each)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Comparison for Wall {wall}')
    plt.legend()
    plt.show()

# 提取 CSV 文件中的 QSURF
q_surf = data['QSURF']

# 计算 Q_wall-zone 的总和
q_wall_zone_sum = q_wall_zone_all.sum(axis=1)
q_wall_zone_sum_measure = q_wall_zone_measure_all.sum(axis=1)

# 绘制 Q_wall-zone 和 QSURF 比较图
plt.figure(figsize=(12, 6))
plt.plot(q_wall_zone_sum, label='Q_wall-zone (Simulated)', linestyle='-', alpha=0.7)#, marker='o'
plt.plot(q_wall_zone_sum_measure, label='Q_wall-zone (Measured)', linestyle='-', alpha=0.7)#, marker='s'
plt.plot(q_surf, label='Q_SURF (Measured)', linestyle='--', alpha=0.7)
plt.xlabel('Time Steps (0.5h each)')
plt.ylabel('Heat Flux (kJ/h)')
plt.title('Comparison of Q_wall-zone and Q_SURF')
plt.legend()
plt.show()

# 保存 RC 参数和比较结果
rc_params_df = pd.DataFrame(rc_params, index=['Rex', 'Rin', 'C']).T
rc_params_df.to_csv('rc_params_curvefit.csv', index=True)
comparison_df = pd.DataFrame({'Q_wall-zone (kJ/h)': q_wall_zone_sum, 'Q_SURF (kJ/h)': q_surf})
comparison_df.to_csv('comparison_curvefit.csv', index=False)
# 将 q_wall_zone_all 数据输出到 CSV 文件
q_wall_zone_all.to_csv('q_wall_zone_all.csv', index=True, header=True)
print("q_wall_zone_all 已保存到 q_wall_zone_all.csv")

print("RC 参数和比较结果已保存。")
