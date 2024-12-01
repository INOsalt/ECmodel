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
def grey_box_model(C_zone, time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow):
    #R_ext_wall: 12.71448996633792 R_zone_wall: 5.636322198546022 C: 10302.175649491874
    R_ext_wall = 12.71448996633792
    R_zone_wall = 5.636322198546022
    c_air = 1005  # 空气比热容 (J/kg·K)
    dt = (time[1] - time[0]) * 3600  # 时间步长 (秒)

    # 初始化变量
    internal_temp = np.zeros(len(time))
    internal_temp[0] = 20.5  # 初始室内空气温度

    # 预计算常量
    R_zone_wall_inv = 1 / R_zone_wall

    for t in range(1, len(time)):
        # 墙体热流
        Q_zone_wall = R_zone_wall_inv * (wall_temp[t] - internal_temp[t - 1])

        # 空间采暖与制冷负荷
        Q_space_heat = Q_heat[t]
        Q_space_cool = -Q_cool[t]

        # 内部热负荷
        Q_zone_in = Q_in[t]

        # 通风热流
        if vent_flow[t] > 0:
            Q_vent = vent_flow[t] * c_air * (vent_temp[t] - internal_temp[t - 1])
            # print(vent_flow[t])
            # print(vent_temp[t])
            # print(internal_temp[t - 1])
        else:
            Q_vent = 0

        # 室内空气温度动态更新
        d_internal_temp = (Q_zone_wall + Q_space_heat + Q_space_cool + Q_zone_in + Q_vent) / C_zone
        # if t < 100:
        #     print(f"Step {t}:")
        #     print(f"  Q_zone_wall  = {Q_zone_wall}")
        #     print(f"  Q_space_heat = {Q_space_heat}")
        #     print(f"  Q_space_cool = {Q_space_cool}")
        #     print(f"  Q_zone_in    = {Q_zone_in}")
        #     print(f"  Q_vent       = {Q_vent}")
        #     print(f"  Net Heat Flow= {Q_zone_wall + Q_space_heat + Q_space_cool + Q_zone_in + Q_vent}")
        #     print(f"  d_internal_temp = {d_internal_temp* dt}")

        if np.isnan(d_internal_temp):
            d_internal_temp = 0

        internal_temp[t] = internal_temp[t - 1] + d_internal_temp * dt
        # print(internal_temp[t - 1])
        # print(d_internal_temp)
        # print(internal_temp[t])

    return internal_temp




# 参数辨识目标函数
# 参数辨识目标函数
def objective_function(C_zone, time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, measured_temp):
    """
    目标函数：最小化灰箱模型预测值与测量值之间的均方误差。
    """
    try:
        # 使用当前参数调用灰箱模型
        predicted_temp = grey_box_model(C_zone, time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow)

        # 检查预测结果是否存在 NaN
        if np.any(np.isnan(predicted_temp)):
            print("NaN detected in predicted_temp, assigning high penalty")
            return float('inf')  # 高惩罚值

        # 计算误差 (均方误差)
        mse = mean_squared_error(predicted_temp, measured_temp)

        # 如果 MSE 是 NaN，同样返回高惩罚值
        if np.isnan(mse):
            print("NaN detected in MSE, assigning high penalty")
            return float('inf')

        return mse

    except Exception as e:
        # 捕获异常并返回高惩罚值
        print(f"Exception in objective_function: {e}, assigning high penalty")
        return float('inf')

# 参数辨识方法
def parameter_identification(time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, internal_temp, vent_temp, vent_flow,bounds):

    bounds = bounds
    def ga_objective(C_zone):
        try:
            # 调用目标函数
            return objective_function(C_zone, time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, internal_temp)
        except Exception as e:
            print(f"Exception in ga_objective: {e}, assigning high penalty")
            return float('inf')  # 返回高惩罚值

    # 遗传算法配置
    from geneticalgorithm import geneticalgorithm as ga
    model = ga(
        function=ga_objective,
        dimension=1,  # 参数维度
        variable_type='real',
        variable_boundaries=np.array(bounds),
        algorithm_parameters={
            'max_num_iteration': 500,
            'population_size': 60,
            'mutation_probability': 0.1,
            'elit_ratio': 0.05,
            'parents_portion': 0.3,
            'crossover_probability': 0.8,
            'crossover_type': 'uniform',  # 添加 crossover_type 参数
            'max_iteration_without_improv': 200 #None
        }
    )

    model.run()
    optimized_C_zone = model.output_dict['variable']
    print("优化后的参数：", optimized_C_zone)
    return optimized_C_zone



def visualize_before_correction(time, measured_temp, predicted_temp,iteration):
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
    plt.savefig(f"before_correction {iteration}.png", dpi=300)
    # plt.show()


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
def block_plots():
    plt.show = lambda *args, **kwargs: None  # Override plt.show to do nothing

# 主函数
def main():

    block_plots() #关闭图片显示

    file_path = 'RC.csv'  # 替换为实际文件路径
    results_file = 'results.txt'  # 输出结果文件

    with open(results_file, 'w') as f:
        f.write("灰箱模型参数优化与预测结果\n")
        f.write("=" * 50 + "\n")

    bounds = [[(10, 1e10)],[(10, 1e11)],[(10, 1e12)],[(10, 1e13)],[(10, 1e14)],[(10, 1e15)]]

    for bound in bounds:
        print(f"开始 {bound} ...")

        # 数据准备
        time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, internal_temp = load_real_data(
            file_path)

        # 参数辨识
        C_zone = parameter_identification(
            time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, internal_temp, vent_temp, vent_flow,bound)
        print(f" {bound} 优化后的参数：", C_zone)

        # 灰箱模型预测
        predicted_temp = grey_box_model(
            C_zone, time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow)

        # 评价指标计算
        mse = np.mean((predicted_temp - internal_temp) ** 2)
        print(f"第 {bound} 均方误差（MSE）：{mse:.4f}")

        # 保存结果到txt文件
        with open(results_file, 'a') as f:
            f.write(f" {bound} 运行结果：\n")
            f.write(f"优化后的参数：{C_zone}\n")
            f.write(f"均方误差（MSE）：{mse:.4f}\n")
            f.write("-" * 50 + "\n")

        # 可视化灰箱模型预测与实测值的对比（高斯校正之前）
        visualize_before_correction(time, internal_temp, predicted_temp,bound)

        # 高斯过程校正
        # corrected_temp, gp_model = gaussian_process_correction(time, external_temp, internal_temp, predicted_temp)
        # visualize_and_save_results(time, internal_temp, predicted_temp, corrected_temp, gp_model, filename=f"gp_model_{iteration}.pkl")

    print(f"所有结果已保存到 {results_file} 文件中。")
    """
def main():
    file_path = 'RC.csv'  # 替换为实际文件路径
    results_file = 'results.txt'  # 输出结果文件

    with open(results_file, 'w') as f:
        f.write("灰箱模型参数优化与预测结果\n")
        f.write("=" * 50 + "\n")

    for iteration in range(1, 11):
        print(f"开始第 {iteration} 次运行...")

        # 数据准备
        time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, internal_temp = load_real_data(
            file_path)

        # 参数辨识
        bounds = [(10, 1e12)]
        C_zone = parameter_identification(
            time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, internal_temp, vent_temp, vent_flow,bounds)
        print(f"第 {iteration} 次优化后的参数：", C_zone)

        # 灰箱模型预测
        predicted_temp = grey_box_model(
            C_zone, time, external_temp, wall_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow)

        # 评价指标计算
        mse = np.mean((predicted_temp - internal_temp) ** 2)
        print(f"第 {iteration} 次均方误差（MSE）：{mse:.4f}")

        # 保存结果到txt文件
        with open(results_file, 'a') as f:
            f.write(f"第 {iteration} 次运行结果：\n")
            f.write(f"优化后的参数：{C_zone}\n")
            f.write(f"均方误差（MSE）：{mse:.4f}\n")
            f.write("-" * 50 + "\n")

        # 可视化灰箱模型预测与实测值的对比（高斯校正之前）
        visualize_before_correction(time, internal_temp, predicted_temp,iteration)

        # 高斯过程校正
        # corrected_temp, gp_model = gaussian_process_correction(time, external_temp, internal_temp, predicted_temp)
        # visualize_and_save_results(time, internal_temp, predicted_temp, corrected_temp, gp_model, filename=f"gp_model_{iteration}.pkl")

    print(f"所有结果已保存到 {results_file} 文件中。")
 """

if __name__ == "__main__":
    main()
