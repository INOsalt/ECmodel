import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import differential_evolution
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from standard_class import StandardizedGP




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



# 完整灰箱模型
def grey_box_model(params, time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow):
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
    wall_temp = np.zeros(len(time))  # 墙体温度
    internal_temp = np.zeros(len(time))  # 室内空气温度
    wall_temp[0] = 20  # 初始墙体温度
    internal_temp[0] = 20  # 初始室内空气温度

    for t in range(1, len(time)):
        # 1. 计算墙体与室外的热流 (Q_ext_wall)
        Q_ext_wall = (1 / R_ext_wall) * (external_temp[t] - wall_temp[t - 1])

        # 2. 计算墙体与室内的热流 (Q_zone_wall)
        Q_zone_wall = (1 / R_zone_wall) * (wall_temp[t - 1] - internal_temp[t - 1])

        # 3. 墙体温度动态更新
        d_wall_temp = (Q_ext_wall - Q_zone_wall) / C_wall
        wall_temp[t] = wall_temp[t - 1] + d_wall_temp * dt

        # 4. 直接使用空间负荷
        Q_zone_rad = Q_heat[t]  # 直接等于输入的 Q_heat
        Q_space_cool = - Q_cool[t]

        # 5. 内部热负荷 Q_in
        Q_zone_in = Q_in[t]  # 直接等于输入的 Q_in

        # 6.计算通风系统热流 (Q_vent)T 出口-入口
        temp_diff = vent_temp[t] - internal_temp[t-1]
        # temp_diff = np.clip(temp_diff, -50, 50)  # 限制温差范围在 -50 到 50 之间
        a = vent_temp[t]
        b = internal_temp[t-1]
        if vent_flow[t] > 0:  # 仅在通风质量流量为正时计算
            Q_vent = vent_flow[t] * c_air * temp_diff
        else:
            Q_vent = 0  # 通风流量为负时，设置为 0

        # 7. 室内空气温度动态更新
        d_internal_temp = (Q_zone_wall + Q_zone_rad + Q_vent + Q_zone_in + Q_space_cool) / C_zone  # 将所有热流项除以热容
        internal_temp[t] = internal_temp[t - 1] + d_internal_temp * dt

    return internal_temp



# 参数辨识目标函数
def objective_function(params, time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, measured_temp):
    """
    目标函数：最小化灰箱模型预测值与测量值之间的均方误差。
    """
    # 使用当前参数调用灰箱模型
    predicted_temp = grey_box_model(params, time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow)

    # 计算误差 (均方误差)
    mse = mean_squared_error(predicted_temp, measured_temp)
    return mse


# 参数辨识方法
def parameter_identification(time, external_temp, Q_heat ,Q_cool, Q_in, internal_temp, vent_temp, vent_flow):
    """
    参数辨识：通过最小化误差优化模型参数。
    输入：
    - time: 时间序列 (小时)
    - external_temp: 室外温度 (°C)
    - Q_heat: 空间热负荷 (W)
    - internal_temp: 实测的室内温度 (°C)
    - vent_temp: 通风供气温度 (°C)
    - vent_flow: 通风质量流量 (kg/s)

    输出：
    - 优化后的参数 [R_ext_wall, R_zone_wall, C_wall, C_zone]
    """
    # 初始参数值 [R_ext_wall, R_zone_wall, C_wall, C_zone]
    initial_params = [1, 1, 100, 1000]

    # 参数取值范围
    bounds = [
        (0.01, 1e7),  # R_ext_wall
        (0.01, 1e7),  # R_zone_wall
        (10, 1e7),  # C_wall
        (10, 1e7)  # C_zone
    ]

    # # 调用优化方法
    # result = minimize(
    #     objective_function,  # 目标函数
    #     initial_params,  # 初始参数
    #     args=(time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, internal_temp),  # 其他参数
    #     bounds=bounds,  # 参数范围
    #     method='L-BFGS-B'  # 优化算法
    # )
    # result = differential_evolution(
    #     objective_function,
    #     bounds=bounds,
    #     args=(time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, internal_temp)
    # )
    # 全局优化（差分进化）
    global_result = differential_evolution(
    objective_function,
    bounds=bounds,
    args=(time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, internal_temp),
    strategy='best1bin',
    maxiter=1000,
    popsize=20)

    # 局部优化（L-BFGS-B）
    result = minimize(
        objective_function,
        global_result.x,
        args=(time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, internal_temp),
        bounds=bounds,
        method='L-BFGS-B'
    )

    # 返回优化后的参数
    return result.x


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


# 高斯过程校正
def gaussian_process_correction(time, external_temp, measured_temp, predicted_temp,
                                initial_downsample_rate=1, max_retries=10):
    """
    使用 StandardizedGP 类实现高斯过程回归校正灰箱模型的误差，并处理可能的收敛问题。

    参数:
    - time: 时间序列 (小时, 一维数组)
    - external_temp: 室外温度 (°C, 一维数组)
    - measured_temp: 实测室内温度 (°C, 一维数组)
    - predicted_temp: 灰箱模型预测的室内温度 (°C, 一维数组)
    - initial_downsample_rate: 初始降采样率，用于减少训练数据规模 (默认: 1, 不降采样)
    - max_retries: 最大重试次数，用于调整参数解决收敛问题

    返回:
    - corrected_temp: 校正后的室内温度 (°C, 一维数组)
    - gp_model: 训练好的 StandardizedGP 模型对象
    """
    residual = measured_temp - predicted_temp  # 计算残差（目标值）
    X = np.column_stack((time, external_temp))  # 构建输入特征

    downsample_rate = initial_downsample_rate  # 设置初始降采样率
    retry_count = 0  # 当前重试次数

    while retry_count <= max_retries:
        try:
            # 初始化 StandardizedGP 模型，设置当前降采样率
            gp_model = StandardizedGP(
                kernel=C(1.0, (1e-2, 1e2)) * RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e3)),
                n_restarts_optimizer=10,
                alpha=1e-2,
                downsample_rate=downsample_rate
            )

            # 训练模型
            gp_model.fit(X, residual)

            # 训练成功，打印核参数
            print(f"高斯过程收敛成功！降采样率: {downsample_rate}, 核参数: {gp_model.gp.kernel_}")

            # 使用模型预测校正值
            correction = gp_model.predict(X)
            corrected_temp = predicted_temp + correction

            return corrected_temp, gp_model

        except Exception as e:
            # 捕获收敛失败的异常
            print(f"高斯过程收敛失败，重试中 (尝试次数: {retry_count + 1}/{max_retries})")
            print(f"异常信息: {e}")

            # 增大降采样率以减少训练数据规模
            downsample_rate *= 2
            print(f"调整降采样率至: {downsample_rate}")

            # 调整模型参数：增大核函数的长度尺度
            if retry_count == max_retries - 1:  # 最后一次重试
                print("尝试更改高斯过程模型参数...")
                gp_model.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=20.0, length_scale_bounds=(1e-1, 1e3))

            retry_count += 1

    # 如果最终仍未收敛，返回未校正值
    print("高斯过程校正失败，返回未校正的预测值。")
    return predicted_temp, None


def visualize_and_save_results(time, measured_temp, predicted_temp, corrected_temp, model, filename="gp_model.pkl"):
    """
    Visualize results and save the trained Gaussian Process model to a file.

    Parameters:
    - time: Time sequence (hours)
    - measured_temp: Measured indoor temperature
    - predicted_temp: Predicted indoor temperature by the grey-box model
    - corrected_temp: Corrected indoor temperature by Gaussian Process
    - model: Trained Gaussian Process model
    - filename: Name of the file to save the model (default: "gp_model.pkl")
    """
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(time, measured_temp, label="Measured Temperature", linewidth=2)
    plt.plot(time, predicted_temp, label="Grey-box Model Prediction", linestyle='--', linewidth=2)
    plt.plot(time, corrected_temp, label="Corrected Prediction", linestyle='-.', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Indoor Temperature (°C)")
    plt.legend()
    plt.title("Indoor Temperature Prediction and Correction")
    plt.grid(True)
    plt.savefig("results.png", dpi=300)
    plt.show()

    # Save the trained Gaussian Process model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# 主函数
def main():
    # 数据准备

    file_path = 'RC.csv'  # 替换为实际文件路径
    time, external_temp, wall_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow, internal_temp = load_real_data(file_path)
    #
    # 参数辨识
    params = parameter_identification(time, external_temp, Q_heat ,Q_cool, Q_in, internal_temp, vent_temp, vent_flow)
    print("优化后的参数：", params)

    # params = [100,99.99988063,10000,1000.00022737]
    # params = [241.11551383,206.98652076,1000.,36.33634168]
    #[ 227.97206597  884.04197974 9916.2309157  9962.6445412 ]
    # 灰箱模型预测
    predicted_temp = grey_box_model(params, time, external_temp, Q_heat ,Q_cool, Q_in, vent_temp, vent_flow)

    # 可视化灰箱模型预测与实测值的对比（高斯校正之前）
    visualize_before_correction(time, internal_temp, predicted_temp)
    #
    # # 高斯过程校正
    # corrected_temp, gp_model = gaussian_process_correction(time,external_temp, internal_temp, predicted_temp)

    # # 可视化结果
    # visualize_and_save_results(time, internal_temp, predicted_temp, corrected_temp, gp_model, filename="gp_model.pkl")


if __name__ == "__main__":
    main()
