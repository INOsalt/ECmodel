import numpy as np
import math
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import differential_evolution
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from standard_class import StandardizedGP
from geneticalgorithm import geneticalgorithm as ga
from thermal_model import ThermalModel


def load_real_data(file_path):
    """
    从实际数据集加载数据并进行单位转换。

    输入:
    - file_path: 数据文件路径 (CSV)

    输出:
    - time: 时间序列 (小时)
    - external_temp: 室外温度 (°C)
    - Q_heat: 空间热负荷 (W)
    - vent_temp: 通风供气温度 (°C)
    - vent_flow: 通风质量流量 (kg/s)
    - measured_temp: 实测室内温度 (°C)
    """
    # 读取数据
    df = pd.read_csv(file_path)

    # 确认时间间隔为 0.5 小时，生成时间序列 (小时)
    time = df['TIME'].values * 0.5  # 原始数据以 0.5 为步长

    # 室外温度 (°C)
    external_temp = df['Tout'].values

    # 墙温
    wall_temp = df['Twall'].values

    # 空间热负荷 (kJ/hr -> W)
    # 注意: 1 kJ/hr = 0.2778 W
    Q_heat = df['QHEAT_Zone1'].values * 0.2778
    Q_cool = df['QCOOL_Zone1'].values * 0.2778
    # Q_in = np.zeros(len(df['QHEAT_Zone1']))
    Q_in = df['Qin_kJph'].values * 0.2778

    # 通风供气温度 (°C)
    vent_temp = df['TAIR_fresh'].values

    # 通风流量 (kg/hr -> kg/s)
    # 注意: 1 kg/hr = 1/3600 kg/s
    vent_flow = df['Mrate_kgph'].values / 3600

    # 实测室内温度 (°C)
    measured_temp = df['TAIR_Zone1'].values

    return time, external_temp,wall_temp, Q_heat,Q_cool, Q_in, vent_temp, vent_flow, measured_temp

def Q_model(params, time, external_temp, measured_temp, wall_temp, vent_temp, vent_flow):
    """
    完整热平衡灰箱模型实现，直接使用散热器的空间热负荷 Q_heat。

    参数:
    - params: 模型参数 [R_ext_wall, R_zone_wall, C_wall, C_zone]
    - time: 时间序列 (小时)
    - external_temp: 室外温度 (°C)
    - Q_heat: 空间热负荷 (W)
    - vent_temp: 通风供气温度 (°C)
    - vent_flow: 通风质量流量 (kg/s)

    返回:
    - internal_temp: 室内空气温度的时间序列 (°C)
    """
    R_ext_wall, R_zone_wall, C_wall, C_zone = params
    c_air = 1005  # 空气比热容 (J/kg·K)
    dt = time[1] - time[0]  # 时间步长 (小时)

    # 初始化变量

    internal_temp = measured_temp  # 室内空气温度
    Q_zone_wall_list = np.zeros(len(time))
    Q_vent_list = np.zeros(len(time))


    for t in range(1, len(time)):
        # 1. 计算墙体与室外的热流 (Q_ext_wall)
        Q_ext_wall = (1 / R_ext_wall) * (external_temp[t] - wall_temp[t - 1])

        # 2. 计算墙体与室内的热流 (Q_zone_wall)
        Q_zone_wall = (1 / R_zone_wall) * (wall_temp[t - 1] - internal_temp[t - 1])

        # 3. 墙体温度动态更新
        d_wall_temp = (Q_ext_wall - Q_zone_wall) / C_wall
        wall_temp[t] = wall_temp[t - 1] + d_wall_temp * dt

        # 6.计算通风系统热流 (Q_vent)T 出口-入口
        temp_diff = vent_temp[t] - internal_temp[t-1]
        if vent_flow[t] > 0:  # 仅在通风质量流量为正时计算
            Q_vent = vent_flow[t] * c_air * temp_diff
        else:
            Q_vent = 0  # 通风流量为负时，设置为 0
        Q_zone_wall_list[t] = Q_zone_wall
        Q_vent_list[t] = Q_vent


    return  Q_zone_wall_list, Q_vent_list

# 参数辨识目标函数
def objective_function(params, time, external_temp,wall_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, measured_temp):
    """
    目标函数：最小化灰箱模型预测值与测量值之间的均方误差。
    """
    c_air = 1005  # 空气比热容 (J/kg·K)
    dt = time[1] - time[0]  # 时间步长 (小时)

    # 初始化变量
    thermal_model = ThermalModel(params, gp_model=None)
    Q_zone_wall, Q_ahu = Q_model(params, time, external_temp, measured_temp, wall_temp, vent_temp, vent_flow)
    # measured_Q_obj = abs(Q_zone_wall) + abs(Q_heat) + abs(Q_ahu) + abs(Q_cool)
    # measured_Q_obj = abs(Q_heat) + abs(Q_cool)
    measured_Q_obj = Q_zone_wall + Q_heat + Q_ahu + Q_in - Q_cool


    predicted_Q_obj = np.zeros(len(time))
    predicted_Tin_t1 = np.zeros(len(time))
    predicted_Tin_t1[0] = measured_temp[0]

    for t in range(1, len(time)):
        Tamb_t = external_temp[t]
        Tin_t = predicted_Tin_t1[t-1]
        Qin_t = Q_in[t]
        vent_flow_t = vent_flow[t]
        step_pre = 0.5  # 时间步长为 0.5 小时

        # 调用 predict 方法
        Tin_t1, Twall_t1_pre, T_vent1_pre, Q_zone_pre, Q_ahu_pre, Q_space_heat_pre, Q_space_cool_pre, Q_zone_wall_pre = thermal_model.predict(
            Tamb_t, Tin_t, Qin_t, step_pre, vent_flow_t, Tsp_high=24, Tsp_low=21)  #
        # Q_obj = abs(Q_zone_wall_pre) +  abs(Q_space_heat_pre) + abs(Q_space_cool_pre) + abs(Q_space_cool_pre)
        # Q_obj = abs(Q_space_heat_pre) + abs(Q_space_cool_pre)
        Q_obj = Q_zone_pre
        predicted_Q_obj[t] = Q_obj
        predicted_Tin_t1[t] = Tin_t1


    mse = mean_squared_error(predicted_Q_obj, measured_Q_obj)
    return mse

# 参数辨识方法
def parameter_identification(time, external_temp,wall_temp, Q_heat ,Q_cool, Q_in, internal_temp, vent_temp, vent_flow):
    bounds = [
        (0.1, 1e7),  # R_ext_wall
        (0.1, 1e7),  # R_zone_wall
        (10, 1e7),  # C_wall
        (10, 1e7)  # C_zone
    ]

    def ga_objective(params):
        return objective_function(params, time, external_temp,wall_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, internal_temp)

    model = ga(
        function=ga_objective,
        dimension=4,  # 参数维度
        variable_type='real',
        variable_boundaries=np.array(bounds),
        algorithm_parameters={
            'max_num_iteration': 500,
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'parents_portion': 0.3,
            'crossover_probability': 0.8,
            'crossover_type': 'uniform',  # 添加 crossover_type 参数
            'max_iteration_without_improv': None
        }
    )

    model.run()
    optimized_params = model.output_dict['variable']
    print("优化后的参数：", optimized_params)
    return optimized_params



def visualize_before_correction(time, measured_temp, predicted_temp):
    """
    在高斯过程校正之前可视化灰箱模型预测值与实测值的对比。

    参数:
    - time: 时间序列 (小时)
    - measured_temp: 实测室内温度 (°C)
    - predicted_temp: 灰箱模型预测的室内温度 (°C)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, measured_temp, label="Measured Temperature", linewidth=2)
    plt.plot(time, predicted_temp, label="Predicted Temperature (Grey-box Model)", linestyle='--', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Indoor Temperature (°C)")
    plt.legend()
    plt.title("Grey-box Model Prediction vs Measured Temperature (Before Correction)")
    plt.grid(True)
    # plt.savefig("before_correction.png", dpi=300)
    plt.show()


# 主函数
def main():
    # 数据准备

    file_path = 'RC.csv'  # 替换为实际文件路径
    time, external_temp, wall_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, measured_temp = load_real_data(file_path)
    #
    # # # 参数辨识
    params = parameter_identification(time, external_temp,wall_temp, Q_heat ,Q_cool, Q_in, measured_temp, vent_temp, vent_flow)
    print("优化后的参数：", params)

    # params = [7541743.36346125,   15391.74986294, 9801447.32847309,   13249.43430226]
    # params = [49761.55491753,   35605.80957396, 3929188.11573589, 8146097.71526216]
    #[5.57761967e+04 4.43984054e+04 7.59796341e+06 6.49778645e+03]

    # 灰箱模型预测
    thermal_model = ThermalModel(params, gp_model=None)
    Q_zone_wall, Q_ahu = Q_model(params, time, external_temp, measured_temp, wall_temp, vent_temp, vent_flow)
    # measured_Q_obj = abs(Q_zone_wall) + abs(Q_heat) + abs(Q_ahu) + abs(Q_cool)
    # measured_Q_obj = abs(Q_heat) + abs(Q_cool)
    # measured_Q_zone =
    measured_Q_obj = Q_zone_wall + Q_heat + Q_ahu + Q_in - Q_cool

    predicted_Q_obj = np.zeros(len(time))
    predeiced_Q_zone = np.zeros(len(time))
    predicted_Tin_t1 = np.zeros(len(time))
    predicted_Tin_t1[0] = measured_temp[0]

    for t in range(1, len(time)):
        Tamb_t = external_temp[t]
        Tin_t = predicted_Tin_t1[t-1]
        Qin_t = Q_in[t]
        vent_flow_t = vent_flow[t]
        step_pre = 0.5  # 时间步长为 0.5 小时

        # 调用 predict 方法
        Tin_t1, Twall_t1_pre, T_vent1_pre, Q_zone_pre, Q_ahu_pre, Q_space_heat_pre, Q_space_cool_pre, Q_zone_wall_pre = thermal_model.predict(
            Tamb_t, Tin_t, Qin_t, step_pre, vent_flow_t, Tsp_high=24, Tsp_low=21)  #
        # Q_obj = abs(Q_zone_wall_pre) +  abs(Q_space_heat_pre) + abs(Q_space_cool_pre) + abs(Q_space_cool_pre)
        # Q_obj = abs(Q_space_heat_pre) + abs(Q_space_cool_pre)
        Q_obj = Q_zone_wall_pre + Q_space_heat_pre + Q_ahu_pre + Q_space_cool_pre + Q_in[t]
        predicted_Q_obj[t] = Q_obj
        predeiced_Q_zone[t] = Q_zone_pre
        predicted_Tin_t1[t] = Tin_t1

    # 可视化灰箱模型预测与实测值的对比（高斯校正之前）
    visualize_before_correction(time, measured_Q_obj, predicted_Q_obj)
    visualize_before_correction(time, measured_Q_obj,  predeiced_Q_zone)


if __name__ == "__main__":
    main()
