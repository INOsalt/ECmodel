import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from parameter_iden import grey_box_model,objective_function,parameter_identification,gaussian_process_correction,visualize_and_save_results



def generate_test_data():
    """
    生成小规模的人工测试数据 (24小时，步长为1小时)。
    数据包括：
    - 时间序列: 24个时间点 (小时)
    - 室外温度: 周期波动的模拟数据
    - 热负荷: 随机波动的热负荷数据
    - 通风温度: 稍高于室外温度，带随机噪声
    - 通风流量: 固定流量，带随机噪声
    - 实测室内温度: 稍高于通风温度，带随机噪声
    """
    time = np.linspace(0, 24, 24)  # 时间序列 (24小时)
    external_temp = 10 + 5 * np.sin(2 * np.pi * time / 24)  # 室外温度 (周期波动)
    Q_heat = 2000 + 200 * np.random.randn(len(time))  # 热负荷 (随机波动)
    vent_temp = external_temp + 2 + 0.5 * np.random.randn(len(time))  # 通风温度 (室外温度稍高)
    vent_flow = 0.05 + 0.005 * np.random.randn(len(time))  # 通风流量 (随机波动)
    measured_temp = vent_temp + 2 + 0.2 * np.random.randn(len(time))  # 室内温度 (稍高于通风温度)

    return time, external_temp, Q_heat, vent_temp, vent_flow, measured_temp


# 主函数
def main():
    # 数据准备

    time, external_temp, radiator_power, vent_temp, vent_flow, internal_temp = generate_test_data()

    # 参数辨识
    params = parameter_identification(time, external_temp, internal_temp, radiator_power, vent_temp, vent_flow)
    print("优化后的参数：", params)

    # 灰箱模型预测
    predicted_temp = grey_box_model(params, time, external_temp, radiator_power, vent_temp, vent_flow)

    # 高斯过程校正
    corrected_temp, gp_model = gaussian_process_correction(time,external_temp, internal_temp, predicted_temp)

    # 可视化结果
    visualize_and_save_results(time, internal_temp, predicted_temp, corrected_temp, gp_model, filename="gp_model.pkl")


if __name__ == "__main__":
    main()
