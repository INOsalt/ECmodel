import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from standard_class import StandardizedGP
from geneticalgorithm import geneticalgorithm as ga

# 读取CSV文件
file_path = 'RCB.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 提取墙壁温度列和室内外温度列
wall_temp_columns = ['TSI_S4', 'TSI_S6',
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10', 'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'TSI_S1', 'TSI_S2', 'TSI_S3',  'TSI_S5',# ext wall
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
def star_model(t, Rstar, Rair, C_air, T_air_ini):
    T_star_simulated = [T_air_ini]  # 初始化模拟温度
    T_air_simulated = [T_air_ini]
    for i in range(1, len(t)):
        T_air_t = T_air_simulated[- 1]
        T_star_t = T_star_simulated[-1]
        dT_star = 0

        for wall in wall_temp_columns:
            T_wall_in = data[wall].values
            T_wall_t = T_wall_in[-1]
            dT_star_temp = dt/C * ((T_wall_t - T_star_t) / Rstar - (T_star_t - T_air_t) / Rair)
            dT_star = dT_star + dT_star_temp
        T_star_simulated.append(T_star_t + dT_star)

        # 空间负荷
        Q_space_t = Q_space[i]  # 直接等于输入的 Q_heat
        # 通风系统热流 (Q_vent)
        temp_diff = vent_temp[i] - T_air_simulated[- 1]
        # temp_diff = np.clip(temp_diff, -50, 50)  # 限制温差范围在 -50 到 50 之间
        try:
            Q_vent_t = vent_flow * c_air * temp_diff
        except OverflowError:
            print(f"Overflow encountered at t={i}, setting Q_vent to 0.")
            Q_vent_t = 0
        Q_in_t = Q_in[i]
        Q_air = (T_star_t - T_air_t) / Rair + Q_space_t + Q_vent_t + Q_in_t
        dT_air = dt / C_air * Q_air
        T_air_simulated.append(T_air_t + dT_air)

    return np.array(T_air_simulated)



# 参数辨识目标函数
def objective_function(params,t,measured_temp):
    """
    目标函数：最小化灰箱模型预测值与测量值之间的均方误差。
    """
    # 使用当前参数调用灰箱模型
    Rstar, Rair, C_air = params
    predicted_temp = star_model(t, Rstar, Rair, C_air, measured_temp[0])
    #
    # if np.any(np.isnan(predicted_temp)) or np.any(np.isnan(measured_temp)):
    #     raise ValueError("Predicted or measured temperature contains NaN values.")
    # if np.any(np.isinf(predicted_temp)) or np.any(np.isinf(measured_temp)):
    #     raise ValueError("Predicted or measured temperature contains infinite values.")

    # 计算误差 (均方误差)
    mse = mean_squared_error(np.nan_to_num(predicted_temp), np.nan_to_num(measured_temp))
    return mse

# 参数辨识方法
def parameter_identification(time, measured_temp):
    bounds = [
        (0.0001, 50),  # R_ext_wall
        (0.0001, 50),  # R_zone_wall
        (10, 1e9),  # C_wall
    ]

    def ga_objective(params):
        return objective_function(params,time,measured_temp)

    model = ga(
        function=ga_objective,
        dimension=3,  # 参数维度
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
# 主函数
def main():

    # # 参数辨识
    T_air_measured = data['Tin'].values
    time = np.arange(len(T_air_measured))  # 时间步数
    params = parameter_identification(time, T_air_measured)
    print("优化后的参数：", params)

    Rstar, Rair, C_air = params

    # 灰箱模型预测
    T_air_predicted = star_model(time, Rstar, Rair, C_air, T_air_measured[0])

    # 可视化灰箱模型预测与实测值的对比（高斯校正之前）
    visualize_before_correction(time, T_air_measured, T_air_predicted)
    #
    #
    # # 高斯过程校正
    # corrected_temp, gp_model = gaussian_process_correction(time,external_temp, internal_temp, predicted_temp)
    #
    # # 可视化结果
    # visualize_and_save_results(time, internal_temp, predicted_temp, corrected_temp, gp_model, filename="gp_model.pkl")


if __name__ == "__main__":
    main()
