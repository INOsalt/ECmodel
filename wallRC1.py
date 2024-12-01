import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'RCB.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 提取墙壁温度列和室内外温度列
wall_temp_columns = ['TSI_S7', 'TSI_S8', 'TSI_S9','TSI_S10', 'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']


# 时间步长
dt = 1800  # 0.5小时 -> 秒

# 定义 RC 模型
def rc_model(params, T_wall_ext, T_air, T_wall_int_measured, dt):
    Rex, Rin, C = params
    T_wall_int_simulated = [T_wall_int_measured[0]]  # 初始化模拟温度
    for t in range(1, len(T_wall_ext)):
        T_int_t = T_wall_int_simulated[-1]
        dT = dt / C * ((T_wall_ext[t-1] - T_int_t) / Rex - (T_int_t - T_air[t-1]) / Rin)
        T_wall_int_simulated.append(T_int_t + dT)
    return np.array(T_wall_int_simulated)

# 定义损失函数
def loss_function(params, T_wall_ext, T_air, T_wall_int_measured, dt):
    T_wall_int_simulated = rc_model(params, T_wall_ext, T_air, T_wall_int_measured, dt)
    return np.sum((T_wall_int_measured - T_wall_int_simulated) ** 2)

# 初始化结果存储
rc_params = {}
q_wall_zone_all = pd.DataFrame(index=data.index)

# 针对每面墙拟合 RC 参数
for wall in wall_temp_columns:
    T_wall_int_measured = data[wall].values
    T_wall_ext = data['Tout'].values  # 墙体外表面温度
    T_air = data['Tin'].values  # 室内空气温度

    # 初始参数猜测和优化
    initial_guess = [0.001, 0.001, 10000]  # 初始值
    result = minimize(
        loss_function, initial_guess, args=(T_wall_ext, T_air, T_wall_int_measured, dt),
        method='L-BFGS-B', bounds=[(0.001, 1000),(0.001, 1000), (100, 10000000)]
    )
    Rex_opt, Rin_opt, C_opt = result.x
    rc_params[wall] = (Rex_opt, Rin_opt, C_opt)


    # 计算每个时间步长的 Q_wall-zone (单位转换为 kJ/h)
    T_wall_int_simulated = rc_model([Rex_opt, Rin_opt, C_opt], T_wall_ext, T_air, T_wall_int_measured, dt)
    plt.plot(T_wall_int_measured, label='Measured', linestyle='--', alpha=0.7)
    plt.plot(T_wall_int_simulated, label='Simulated', linestyle='-', alpha=0.7)
    plt.legend()
    plt.show()
    q_wall_zone = (T_wall_int_simulated - T_air) / Rin_opt * dt * (3600 / 1000)
    q_wall_zone_all[wall] = q_wall_zone

# 提取 CSV 文件中的 QSURF
q_surf = data['QSURF']

# 计算 Q_wall-zone 的总和
q_wall_zone_sum = q_wall_zone_all.sum(axis=1)

# 绘制比较图
plt.figure(figsize=(12, 6))
plt.plot(q_wall_zone_sum, label='Q_wall-zone (Simulated)', linestyle='-', alpha=0.7)
plt.plot(q_surf, label='Q_SURF (Measured)', linestyle='--', alpha=0.7)
plt.xlabel('Time Steps (0.5h each)')
plt.ylabel('Heat Flux (kJ/h)')
plt.title('Comparison of Q_wall-zone and Q_SURF')
plt.legend()
plt.show()

# 保存 RC 参数和比较结果
rc_params_df = pd.DataFrame(rc_params, index=['Rex', 'Rin', 'C']).T
rc_params_df.to_csv('rc_params.csv', index=True)
comparison_df = pd.DataFrame({'Q_wall-zone (kJ/h)': q_wall_zone_sum, 'Q_SURF (kJ/h)': q_surf})
comparison_df.to_csv('comparison.csv', index=False)

print("RC 参数和比较结果已保存。")
