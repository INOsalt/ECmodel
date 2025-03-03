import numpy as np
import pandas as pd
import joblib
from joblib import load

class ThermalModel:
    def __init__(self, params, wall_RC_params, gp_model):
        """
        初始化 ThermalModel 类

        参数:
        - params: 预训练的 2R2C 模型参数 [R_ext_wall, R_zone_wall, C_wall, C_zone]
        - gp_model: 高斯过程模型，用于室温预测误差校正
        """
        self.gp_model = gp_model
        self.c_air = 1005  # 空气比热容 (J/kg·K)
        self.dt = None  # 时间步长将在后续设置
        self.Rstar_win, self.Rstar_wall, self.Rair, self.Cstar, self.C_air = params
        self.wall_RC = wall_RC_params 
        self.wall_temp_columns = ['TSI_S4', 'TSI_S6',#roof
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10',#window
                     'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'TSI_S1', 'TSI_S2', 'TSI_S3',  'TSI_S5',# ext wall


    def predict_next(self, Tamb_t, Tin_t,Twall_t_dict,Tstar_t, Qin_t, step_pre, vent_flow, Tsp_high=24,Tsp_low=21):
        """
        预测下一时刻的热负荷和温度

        参数:
        - Tamb_t: 当前环境温度 (°C)
        - Tin_t: 当前室内温度 (°C)
        - Qin_t: 当前内部热负荷 (W)
        - step_pre: 当前预测的时间步长 (小时)
        - vent_flow: 通风质量流量 (kg/s)

        返回:
        - Tin_t1: 下一时刻室内温度 (校正后的) (°C)
        - Twall_t1: 下一时刻墙体温度 (°C)
        - Q_zone: 室内热平衡负荷 (W)
        - Q_ahu: AHU 负荷 (W)
        - Q_space_heat: 空间加热负荷 (W)
        - Q_space_cool: 空间制冷负荷 (W)
        """
        self.dt = step_pre  # 设置时间步长

        Qwall_star_t = 0
        Twall_t1_dict = {}
        for wall in self.wall_temp_columns:
            Twall_t = Twall_t_dict[wall]
            Rex, Cwall = self.wall_RC.loc[wall, ['Rex', 'C']]
            if wall in ['TSI_S4', 'TSI_S6']:
                Tamb_t = Twall_t
            if wall in ['TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10']:
                Rstar = self.Rstar_win
            else:
                Rstar = self.Rstar_wall
            dTwall = self.dt / Cwall * ((Tamb_t - Twall_t) / Rex - (Twall_t - Tstar_t) / Rstar)
            Twall_t1 = Twall_t + dTwall
            Twall_t1_dict[wall] = Twall_t1
            Qwall_star_temp = (Twall_t - Tstar_t) / Rstar
            Qwall_star_t = Qwall_star_t + Qwall_star_temp
        
        dTstar = self.dt/self.Cstar * (Qwall_star_t - (Tstar_t - Tin_t) / self.Rair)
        Tstar_t1 = Tstar_t + dTstar

        Tin_t1 = Tsp_high
        dTin = (Tin_t1 - Tin_t)/self.dt
        # **计算下一时刻室内温度**
        Qzone_t = dTin * self.C_air

        # **计算通风热流 (AHU 负荷)**
        Tsp_vent = self._compute_Tsp_vent(Tin_t)
        T_vent = Tsp_vent + 0.5
        temp_diff = T_vent - Tin_t
        Qahu_t = vent_flow * self.c_air * temp_diff

        #Q surface
        Qstar_t = (Tstar_t - Tin_t) / self.Rair
        # **热平衡
        Qspace_t = Qzone_t - Qahu_t - Qstar_t - Qin_t

        # **计算空间加热和制冷负荷**
        if Qspace_t > 0:  # 加热
            Tin_t1 = Tsp_low
            dTin = (Tin_t1 - Tin_t)/self.dt
            Qzone_t = dTin * self.C_air
            # **热平衡
            Qspace_t = Qzone_t - Qahu_t - Qstar_t - Qin_t
            Qspace_t = max(Qspace_t,0)
            Qzone1_t = Qspace_t + Qahu_t + Qstar_t + Qin_t
            Tin_t1 = Tin_t + Qzone1_t/self.C_air * self.dt

        elif Qspace_t <= 0:  # 制冷
            Qspace_t = Qspace_t


        return Tin_t1, Twall_t1_dict, T_vent, Tstar_t1, Qzone_t, Qahu_t, Qspace_t

    def _compute_Tsp_vent(self, Tin_t):
        """
        根据规则计算 Tsp_vent

        参数:
        - Tin_t: 当前室内温度 (°C)

        返回:
        - Tsp_vent: 通风供气温度设定点 (°C)
        """
        if Tin_t <= 21:
            return 21.0
        elif Tin_t >= 24:
            return 17.0
        else:
            return -1.333 * Tin_t + 49 - 0.5
    def predict_peiod(self, time_horzion, Tamb_t_list, Tin_t, Qin_t_list, step_pre, vent_flow):
        Twall_t_dict_0 = {}
        for wall in self.wall_temp_columns:
            Twall_t_dict_0[wall] = Tin_t
        Tin_t_list = [Tin_t]
        Tstar_t_0 = Tin_t
        Tstar_t_list = [Tstar_t_0]
        Qzone_t_list = [0]
        Twall_t_dict_list = [Twall_t_dict_0]
        Qahu_t_list = [0]
        Qspace_t_list = [0]
        for i in range(time_horzion): #time horizon步长
            Tin_t = Tin_t_list[i]
            Tamb_t = Tamb_t_list[i]
            Qin_t = Qin_t_list[i]
            Tin_t1, Twall_t1_dict, T_vent, Tstar_t1, Qzone_t1, Qahu_t1, Qspace_t1 = self.predict_next(Tamb_t, Tin_t,Twall_t_dict_0,Tstar_t_0, Qin_t, step_pre, vent_flow)
            Tin_t_list.append(Tin_t1)
            Tstar_t_list.append(Tstar_t1)
            Twall_t_dict_list.append(Twall_t1_dict)
            Qzone_t_list.append(Qzone_t1)
            Qahu_t_list.append(Qahu_t1)
            Qspace_t_list.append(Qspace_t1)
            i += 1
        
        # 模拟加载高斯过程模型
        try:
            gp_model = load("gp_model.pkl")
            print("高斯过程模型已加载。")
            # 构建输入特征
            time = np.arange(len(Tin_t_list)) * step_pre
            external_temp = np.array(Tamb_t_list)
            X = np.column_stack((time, external_temp))

            # 使用模型预测校正值
            correction = gp_model.predict(X)
            Qspace_t_list_corrected = Qspace_t_list + correction
        except FileNotFoundError:
            gp_model = None
            Qspace_t_list_corrected = Qspace_t_list
            print("未找到高斯过程模型，继续使用未校正模型。")
            return Tin_t_list, Twall_t_dict_list, Tstar_t_list, Qzone_t_list, Qahu_t_list, Qspace_t_list, Qspace_t_list_corrected


# 主函数，仅在直接运行脚本时执行
if __name__ == "__main__":
    from joblib import load

    # 模拟加载高斯过程模型
    try:
        gp_model = load("gp_model.pkl")
        print("高斯过程模型已加载。")
    except FileNotFoundError:
        gp_model = None
        print("未找到高斯过程模型，继续使用未校正模型。")

    # 预训练的 2R2C 模型参数
    params = [0.0001, 0.001999, 0.00062838, 40723395.97479104, 200671880.13560498]
    wall_RC_params = pd.read_csv('rc_params_curvefit.csv', index_col=0)
    wall_temp_columns = ['TSI_S4', 'TSI_S6',#roof
                     'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10',#window
                     'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'TSI_S1', 'TSI_S2', 'TSI_S3',  'TSI_S5',# ext wall
    
    # 初始化 ThermalModel 类
    thermal_model = ThermalModel(params, wall_RC_params, gp_model)

    # 输入参数
    Tamb_t = 15  # 当前环境温度
    Tstar_t = 22
    Tin_t = 22  # 当前室内温度
    Qin_t = 100  # 内部热负荷
    vent_flow = 0.2  # 通风质量流量
    step_pre = 0.5  # 时间步长
    Twall_t_dict = {}
    for wall in wall_temp_columns:
        Twall_t_dict[wall] = Tin_t

    # 执行预测
    Tin_t1, Twall_t1_dict, T_vent, Tstar_t1, Qzone_t, Qahu_t, Qspace_t= thermal_model.predict_next(Tamb_t, Tin_t,Twall_t_dict,Tstar_t, Qin_t, step_pre, vent_flow, Tsp_high=24,Tsp_low=21)

    # 输出预测结果
    print(f"下一时刻室温 Tin_t+1: {Tin_t1:.2f}°C")
    print("下一时刻墙体温度 Twall_t1:")
    for wall, temp in Twall_t1_dict.items():
        print(f"  {wall}: {temp:.2f}°C")
    print(f"室内热平衡负荷 Q_zone: {Qzone_t:.2f} W")
    print(f"AHU 负荷 Q_ahu: {Qahu_t:.2f} W")
    print(f"空间负荷 Q_space: {Qspace_t:.2f} W")




