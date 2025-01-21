import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 读取CSV文件
file_path = 'RCB.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 提取墙壁温度列和室内外温度列
wall_temp_columns = ['TSI_S4', 'TSI_S6',#roof
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10',#window
                     'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'TSI_S1', 'TSI_S2', 'TSI_S3',  'TSI_S5',# ext wall
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
#Rair Cair
T_air_measured = data['Tin'].values
T_wall_ext = data['Tout'].values
t = np.arange(len(T_wall_ext))  # 时间步数
# 定义目标函数（遗传算法的适应度函数）
def fitness_function(params):
    Rstar_win, Rstar_wall, C_air = params
    # 模拟空气温度
    T_air_simulated = air_model(t, Rstar_win, Rstar_wall, C_air, T_air_measured[0])
    # 计算均方误差 (MSE) 作为适应度
    mse = np.mean((T_air_measured - T_air_simulated) ** 2)
    return mse,

# 定义遗传算法的设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# 定义基因编码范围 (与参数的 bounds 一致)
param_bounds = [(0.0001, 0.05),   # Rstar_win
                (0.0001, 0.05),    # Rstar_wall
                (10, 1e10)]          # C_air

# 注册遗传算法工具
# 注册基因初始化函数，每个参数范围独立定义
toolbox.register("attr_float", lambda low, up: np.random.uniform(low, up))
toolbox.register("individual", tools.initCycle, creator.Individual,
                 [lambda: np.random.uniform(b[0], b[1]) for b in param_bounds], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
toolbox.register("mutate", tools.mutPolynomialBounded, eta=1.0, low=[b[0] for b in param_bounds],
                 up=[b[1] for b in param_bounds], indpb=0.2)  # 多项式变异
toolbox.register("select", tools.selTournament, tournsize=3)

# 设置种群和遗传算法参数
population = toolbox.population(n=100)  # 种群大小
ngen = 500  # 迭代代数
cxpb = 0.7  # 交叉概率
mutpb = 0.1  # 变异概率

# 执行遗传算法优化
best_params, log = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, verbose=True)

# 提取最佳参数
best_individual = tools.selBest(population, k=1)[0]
Rstar_opt_win, Rstar_opt_wall, Rair_opt, Cstar_opt, C_air_opt = best_individual
print("Optimized Parameters:")
print(f"Rstar_win: {Rstar_opt_win}, Rstar_wall: {Rstar_opt_wall}, Rair: {Rair_opt}, "
      f"Cstar: {Cstar_opt}, C_air: {C_air_opt}")
# 写入到文本文件
file_path = "pso_parameters.txt"  # 你可以根据需要修改文件路径
with open(file_path, "a") as f:  # "a" 模式确保文件不存在时会创建，且内容追加到文件末尾
    f.write("Optimized Parameters:\n")
    f.write(f"Rstar_win: {Rstar_opt_win}, Rstar_wall: {Rstar_opt_wall}, Rair: {Rair_opt}, "
            f"Cstar: {Cstar_opt}, C_air: {C_air_opt}\n")

# 使用最佳参数计算模拟结果
T_air_simulated_cal = air_model(
    t, Rstar_opt_win, Rstar_opt_wall, C_air_opt, T_air_measured[0]
)

# 绘制模拟与实际温度对比
plt.figure(figsize=(8, 4))
plt.plot(T_air_measured, label='Measured', linestyle='--', alpha=0.7)
plt.plot(T_air_simulated_cal, label='Simulated (GA)', linestyle='-', alpha=0.7)
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
