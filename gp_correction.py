from thermal_model import ThermalModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import matplotlib.pyplot as plt
import joblib
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
    Q_vent_list[0] = 21
    predicted_Tin_t = np.zeros(len(time))


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


def gaussian_process_correction(time, external_temp, measured, predicted,
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
    residual = measured - predicted  # 计算残差（目标值）
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
            corrected = predicted + correction

            # Save the trained Gaussian Process model
            joblib.dump(gp_model, "gp_model.pkl")
            print(f"Model saved to gp_model ")
            return corrected, gp_model

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
    return predicted, None

def gaussian_correction(time, external_temp, predicted, gp_model_name):
    """
    使用保存的 StandardizedGP 模型加载并校正预测值。

    参数:
    - time: 时间序列 (小时, 一维数组)
    - external_temp: 室外温度 (°C, 一维数组)
    - predicted: 灰箱模型预测的室内温度 (°C, 一维数组)
    - gp_model_name: 保存的模型文件路径

    返回:
    - corrected: 校正后的室内温度 (°C, 一维数组)
    """
    # 加载保存的模型
    gp_model = joblib.load(gp_model_name)
    print(f"已加载模型: {gp_model_name}")

    # 构建输入特征
    X = np.column_stack((time, external_temp))

    # 使用模型预测校正值
    correction = gp_model.predict(X)
    corrected = predicted + correction

    return corrected

def visualize_before_correction(time, measured_temp, predicted_temp):
    """
    在高斯过程校正之前可视化灰箱模型预测值与实测值的对比。

    参数:
    - time: 时间序列 (小时)
    - measured_temp: 实测室内温度 (°C)
    - predicted_temp: 灰箱模型预测的室内温度 (°C)
    """
    predicted_temp[0]=measured_temp[0]
    plt.figure(figsize=(12, 6))
    plt.plot(time, measured_temp, label="Measured", linewidth=2)
    plt.plot(time, predicted_temp, label="Predicted(Grey-box Model)", linestyle='--', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Indoor Temperature (°C)")
    plt.legend()
    plt.title("Grey-box Model Prediction vs Measured Temperature (Before Correction)")
    plt.grid(True)
    # plt.savefig("before_correction.png", dpi=300)
    plt.show()

def visualize(time, measured_temp, predicted_temp, corrected_temp):
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
    plt.plot(time, measured_temp, label="Measured Q_zone", linewidth=2)
    plt.plot(time, predicted_temp, label="Grey-box Model Prediction", linestyle='--', linewidth=2, alpha = 0.6)
    plt.plot(time, corrected_temp, label="Corrected Prediction", linestyle='-.', linewidth=2, alpha = 0.6)
    plt.xlabel("Time (hours)")
    plt.ylabel("Indoor Temperature (°C)")
    plt.legend()
    plt.title("Indoor Temperature Prediction and Correction")
    plt.grid(True)
    plt.savefig("results.png", dpi=300)
    plt.show()





file_path = 'RC.csv'  # 替换为实际文件路径
time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, measured_temp = load_real_data(file_path)

# 初始化 ThermalModel
# params = [49761.55491753,   35605.80957396, 3929188.11573589, 8146097.71526216]
# params = [7541743.36346125,   15391.74986294, 9801447.32847309,   13249.43430226]
# params = [3846802.33088574, 9991777.67824654, 3515857.99438999, 1897558.09183796]
params = [12.71448996633792, 5.636322198546022, 10302.175649491874, 6.82306117e+09]

Q_zone_wall, Q_vent = Q_model(params, time, external_temp, measured_temp,wall_temp, vent_temp, vent_flow)

measured_Q_zone = Q_zone_wall + Q_heat + Q_vent + Q_in - Q_cool

thermal_model = ThermalModel(params, gp_model=None)

# 运行 2R2C 模型，计算预测的 Q_zone 和室内温度
predicted_Q_zone = np.zeros(len(time))
predicted_Q_ahu  = np.zeros(len(time))
predicted_temp  = np.zeros(len(time))
predicted_Q_cool  = np.zeros(len(time))
predicted_Q_heat  = np.zeros(len(time))
predicted_Twall = np.zeros(len(time))
predicted_T_vent = np.zeros(len(time))

predicted_Tin_t1 = np.zeros(len(time))
predicted_Tin_t1[0] = measured_temp[0]

for t in range(1, len(time)):
    Tamb_t = external_temp[t]
    Tin_t = predicted_Tin_t1[t-1]
    Qin_t = Q_in[t]
    vent_flow_t = vent_flow[t]
    step_pre = 0.5  # 时间步长为 0.5 小时

    # 调用 predict 方法
    Tin_t1, Twall_t1_pre, T_vent1_pre, Q_zone_pre, Q_ahu_pre, Q_space_heat_pre, Q_space_cool_pre, Q_zone_wall_pre \
        = thermal_model.predict(Tamb_t, Tin_t, Qin_t, step_pre, vent_flow_t, Tsp_high=24,Tsp_low=21) #
    predicted_Tin_t1[t] = Tin_t1

    predicted_temp[t] = Tin_t1  # 下一时刻 Tin
    predicted_Q_zone[t] = Q_zone_pre  # Q_zone
    predicted_Q_ahu[t] = Q_ahu_pre  # Q_zone
    predicted_Q_cool[t] = Q_space_cool_pre
    predicted_Q_heat[t] = Q_space_heat_pre
    predicted_Twall[t] = Twall_t1_pre
    predicted_T_vent[t] = T_vent1_pre



#
# #可视化1
visualize_before_correction(time, measured_Q_zone, predicted_Q_zone)
print(measured_temp)
print(predicted_temp)
visualize_before_correction(time, measured_temp, predicted_temp)
# visualize_before_correction(time, wall_temp, predicted_Twall)
visualize_before_correction(time, predicted_T_vent, vent_temp)
visualize_before_correction(time, predicted_Q_ahu, Q_vent)
visualize_before_correction(time, predicted_Q_cool, - Q_cool)

#
# # 训练高斯过程模型
predicted_Q_zone[0] = measured_Q_zone[0]
corrected_Q_zone, gp_model = gaussian_process_correction(time, external_temp, measured_Q_zone, predicted_Q_zone)
#加载模型
# corrected_Q_zone = gaussian_correction(time, external_temp, predicted_Q_zone, "gp_model.pkl")


visualize(time, measured_Q_zone, predicted_Q_zone, corrected_Q_zone)
