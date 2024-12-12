import random
import re
import matplotlib
import scipy.io as sio
import os, time
from copy import deepcopy
from scipy.io import loadmat
import organize_log
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.rcParams['axes.unicode_minus'] = False
# Load the traffic flow data
read_data = pd.read_csv('1.1W_16_daily_728_256_WAI.csv', header=0)
column_name = read_data.iloc[:, 1:-2].columns
# Load the trained model and store the file path of the model parameters (replace according to the actual situation)
origin_model = load_model('daily_cnn_1616_without.h5')
population = np.array([13600000, 11070000, 3769000, 2504000, 682986, 1822000, 6266000, 1609675, 7982000, 17930000, 4085000, 990509,4078000, 2208321, 2896712, 2137000])
#Construct model input (data model)
input_data = read_data.iloc[:, 1:-2]
input_min = input_data.min()
input_max = input_data.max()
delta = input_max - input_min
round_best_var = []
round_best_result = []
set_error = 1e-6
Algorithm_Finished = None
best_var, best_fit = None, None
is_ready = False

global k
global W_16
global start
global ending

def flatten(matrix):
    matrix_flatten = [element[i] for element in matrix for i in range(len(element))]
    return matrix_flatten

## Function normalization
def input_process(origin_input):
    model_in = (origin_input - input_min) / delta
    return model_in

# Constraint input
def g_model_input(ctl_vars):
    global composw2group, g_input
    compose2group = [0 for i in range(5)]
    ctl_vars1 = [0 for i in range(80)]

    if len(ctl_vars) == 80:
        ctl_vars1 = [
            np.array(ctl_vars[:16]),
            np.array(ctl_vars[16:32]),
            np.array(ctl_vars[32:48]),
            np.array(ctl_vars[48:64]),
            np.array(ctl_vars[64:80])
        ]

    for uindex, uitem in enumerate(ctl_vars1):
        compose2group[uindex] = uitem
    model_input = [0 for i in range(5)]
    for group_index, group_item in enumerate(compose2group):
        tmp = flatten_and_divide(group_item * W_16[:, :, ending + 96 + k - 7])
        model_input[group_index] = tmp
    g_input = np.array(model_input)
    return g_input

# Constraint input output
def g_model_output(model_input):
    global total_model_output
    total_model_output = np.array([0])
    for count in range(5):
        model_output = model_runner(model_input[count])
        total_model_output = total_model_output + model_output
    _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=f'模型总输出pso：{total_model_output}')
    with open('log.txt', 'a') as f:
        f.write(str(_))
        f.write('\n')
    return total_model_output

# Particle Swarm Optimization Algorithm
def PSO(x_lb, x_ub, x_dim, max_iter, pop_size, speed_max, w, c1, c2, x_init, daily_resource, D):
    global Algorithm_Finished, best_var, best_fit, round_best_var, round_best_result, g
    # count_g2_max = 15
    r1, r2 = np.random.rand(), np.random.rand()
    speed_lb = [-1 * speed_max for i in range(x_dim)]
    speed_ub = [speed_max for i in range(x_dim)]
    # 种群初始化
    x_dim = int(x_dim)
    pop_size = int(pop_size)
    max_iter = int(max_iter)
    (X_local, X_local_tmp, X_speed, ind_bestpos) = [[[0 for i in range(x_dim)] for j in range(pop_size)] for k in range(4)]
    ind_bestfit = [0 for i in range(pop_size)]
    g_bestpos = [0 for i in range(x_dim)]
    g_bestfit = float("inf")

    for i in range(pop_size):
        if i == 0:
            X_local[i] = x_init
        else:
            X_local[i] = u_history_temp
            X_local[i] = np.array(control_precision(X_local[i]))
        X_local_tmp[i] = X_local[i]
        X_speed[i] = np.array([random.uniform(a, b) for a, b in zip(speed_lb, speed_ub)])

        is_not_satisfied = True
        count_g = 0
        total_model_output = g_model_output(g_model_input(X_local[i]))
        g = total_model_output - daily_resource * (D + 1)
        while is_not_satisfied:
            if g < 0:
                count_g = count_g + 1
                # print("g_count_while2", g)
                if count_g % 10 == 0:
                    r11, r22 = np.random.rand(), np.random.rand()
                    X_speed[i] = w * np.array(X_speed[i]) + c1 * r11 * (
                            np.array(ind_bestpos[i]) - np.array(X_local_tmp[i])) + c2 * r22 * (
                                         np.array(g_bestpos) - np.array(X_local_tmp[i]))
                    for j in range(x_dim):
                        X_speed[i][j] = min(speed_ub[j], X_speed[i][j])
                        X_speed[i][j] = max(speed_lb[j], X_speed[i][j])

                X_local_tmp[i] = [random.uniform(a, b) for a, b in zip(x_lb, x_ub)]
                X_local_tmp[i] = np.array(control_precision(X_local_tmp[i]))
            total_model_output = g_model_output(g_model_input(X_local_tmp[i]))
            g = total_model_output - daily_resource * (D + 1)
            if g >= 0:
                is_not_satisfied = False
                X_local[i] = X_local_tmp[i]
                break
            if count_g >= 30:
                is_not_satisfied = False
                break
        ind_bestpos[i] = X_local[i]
        simulate_parm_dict = ind_bestpos[i]
        ind_bestfit[i] = yield simulate_parm_dict
        if ind_bestfit[i] <= g_bestfit:
            g_bestfit = ind_bestfit[i]
            g_bestpos = ind_bestpos[i]
    del i

    X_fit = [0 for i in range(pop_size)]

    for iter in range(max_iter):
        # #Particle velocity update
        for i in range(pop_size):
            X_speed[i] = w * np.array(X_speed[i]) + c1 * r1 * (np.array(ind_bestpos[i]) - np.array(X_local[i])) + c2 * r2 * (np.array(g_bestpos) - np.array(X_local[i]))
            for j in range(x_dim):
                X_speed[i][j] = min(speed_ub[j], X_speed[i][j])
                X_speed[i][j] = max(speed_lb[j], X_speed[i][j])
        # Particle position update
        for i in range(pop_size):
            X_local_tmp[i] = X_local[i] + X_speed[i]
            X_local_tmp[i] = np.array(control_precision(X_local_tmp[i].tolist()))
            for j in range(x_dim):
                X_local_tmp[i][j] = min(x_ub[j], X_local_tmp[i][j])
                X_local_tmp[i][j] = max(x_lb[j], X_local_tmp[i][j])

            is_not_satisfied = True
            count_g2 = 0
            total_model_output = g_model_output(g_model_input(X_local_tmp[i]))
            g = total_model_output - daily_resource * (D + 1)
            while is_not_satisfied:
                if g < 0:
                    count_g2 = count_g2 + 1
                    if count_g2 % 10 == 0:
                        r11, r22 = np.random.rand(), np.random.rand()
                        X_speed[i] = w * np.array(X_speed[i]) + c1 * r11 * (
                                np.array(ind_bestpos[i]) - np.array(X_local[i])) + c2 * r22 * (
                                             np.array(g_bestpos) - np.array(X_local[i]))
                        for j in range(x_dim):
                            X_speed[i][j] = min(speed_ub[j], X_speed[i][j])
                            X_speed[i][j] = max(speed_lb[j], X_speed[i][j])
                    X_local_tmp[i] = X_local_tmp[i] + X_speed[i]
                    X_local_tmp[i] = np.array(control_precision(X_local_tmp[i].tolist()))
                    for j in range(x_dim):
                        X_local_tmp[i][j] = min(x_ub[j], X_local_tmp[i][j])
                        X_local_tmp[i][j] = max(x_lb[j], X_local_tmp[i][j])
                total_model_output = g_model_output(g_model_input(X_local_tmp[i]))
                g = total_model_output - daily_resource * (D + 1)
                if g >= 0:
                    is_not_satisfied = False
                    X_local[i] = X_local_tmp[i]
                    break
                if count_g2 >= 30:
                    is_not_satisfied = False
                    break

        # fitness update
        for i in range(pop_size):
            simulate_parm_dict = X_local[i]
            X_fit[i] = yield simulate_parm_dict

            if X_fit[i] <= ind_bestfit[i]:
                ind_bestfit[i] = X_fit[i]
                ind_bestpos[i] = X_local[i]

        for i in range(pop_size):
            if ind_bestfit[i] <= g_bestfit:
                g_bestfit = ind_bestfit[i]
                g_bestpos = ind_bestpos[i]

        round_best_var.append(deepcopy(g_bestpos))
        round_best_result.append(deepcopy(g_bestfit))

        _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=f'round_best_var：{round_best_var}')
        with open('log.txt', 'a') as f:
            f.write(str(_))
            f.write('\n')

        _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=f'len： {len(round_best_var)}')
        with open('log.txt', 'a') as f:
            f.write(str(_))
            f.write('\n')

        if len(round_best_var) == 2:
            error = abs(round_best_result[1] - round_best_result[0])
            _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=f"current error:{error}")
            with open('log.txt', 'a') as f:
                f.write(str(_))
                f.write('\n')

            if error != 0:
                if round_best_result[0] < round_best_result[1]:
                    tem = round_best_result[1]
                    round_best_result[1] = round_best_result[0]
                    round_best_result[0] = tem
                if error < set_error:
                    error_achieved = True
                    _ = organize_log.generate_log_msg(success=True, msg_code='I0000',
                                                      msg=f"set_error:{set_error}, current error:{error}, archive set error,quit.")
                    with open('log.txt', 'a') as f:
                        f.write(str(_))
                        f.write('\n')

                    best_var = round_best_result[1]
                    break
            del round_best_result[0]
            del round_best_var[0]

    _ = organize_log.generate_log_msg(success=True, msg_code='I0000',
                                      msg=f"best position:{g_bestpos}, best fitness:{g_bestfit}")
    with open('log.txt', 'a') as f:
        f.write(str(_))
        f.write('\n')

    _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg='Calculation completed！！！ ')
    with open('log.txt', 'a') as f:
        f.write(str(_))
        f.write('\n')

    if isinstance(g_bestpos, list):
        best_var = g_bestpos
    else:
        best_var = g_bestpos.tolist()
    best_fit = round_best_result[0]
    Unused_Var = yield best_var, best_fit, g  # Unused_Var 这个变量没有用   当PSO迭代完成后给出最优适应度和最优变量
    _ = organize_log.generate_log_msg(success=True, msg_code='I0000', \
                                      msg=f"#######################################################" + \
                                          f"Optimal control variable：{best_var} \n" + f"Optimal fitness: {best_fit}" + f'constraint{g}' + \
                                          f"#######################################################")
    with open('log.txt', 'a') as f:
        f.write(str(_))
        f.write('\n')

# Control parameter accuracy
def control_precision(param):
    pattern = re.compile(r'.*?\d+(\.)(\d){6}')
    if isinstance(param, list):
        for indx, item in enumerate(param):
            item = "{:.7f}".format(item)
            if item != str(float("inf")) and item != str(-float('inf')):
                param[indx] = float(re.match(pattern, item).group())
            else:
                pass
    elif isinstance(param, int) or isinstance(param, float):
        param = "{:.7f}".format(param)
        if param != str(float("inf")) and param != str(-float('inf')):
            param = float(re.match(pattern, param).group())
        else:
            pass
    else:
        print(f"Parameter: {param}  {type(param)}")
        print(f"Parameter is illegal")
    return param

# Flatten a two-dimensional matrix into a one-dimensional list
def flatten_and_divide(matrix):
    matrix_flatten = list(matrix.flatten('F'))
    return np.array(matrix_flatten)

# calculate the stage cost of the objective function
def calc_r_sum(matrix, span, end, population):
    result = 0
    for indx in range(end - span + 1, end + 1):
        result = result + np.sum(np.array(matrix[indx]))
        # result = result + np.sum(population * np.array(matrix[indx]))
    return result

def cost_func(a2, x2, p,q):
    f = p * calc_r_sum(a2, 3, 2, population) + q * sum(a2[-1]) + p * calc_r_sum(x2, 3, 2, population) + q * sum(x2[-1])
    return f

# Call the data model and obtain the corresponding wai by inputting UW
def model_runner(external_input):
    input_x = external_input
    model_input = input_process(input_x)
    model_input.fillna(0, inplace=True)
    model_input[model_input == float('inf')] = 0

    y = origin_model.predict(model_input.values.reshape(1, 16, 16))
    output_data = read_data.iloc[:, -1]
    output_sc = MinMaxScaler(feature_range=(0, 1))
    output_sc.fit_transform(output_data.values.reshape(-1, 1))
    model_output = output_sc.inverse_transform(y).reshape(-1)
    return model_output

## ---------------------------------------------------------
#Read the parameters of the mat file
##step1 Load data file
W_16_1 = sio.loadmat('W_16.mat')['W_16']
W_16_initial = W_16_1
W_16 = W_16_initial.reshape(16, 16, 728)
data_population = sio.loadmat('Data_population.mat')

DE_pro_infected = sio.loadmat('DE_pro_infected.mat')['DE_pro_infected']
DE_pro_recovered = sio.loadmat('DE_pro_recovered.mat')['DE_pro_recovered']

I_real_initial = DE_pro_infected  # 2020.03.20 - 2021.06.26 16states  infected population
I_real = I_real_initial.reshape(16, 464)
R_real_initial = DE_pro_recovered  # 2020.03.20 - 2021.06.26
R_real = R_real_initial.reshape(16, 464)
start = 279
ending = 340
L = 16
T = 31
W_16 = W_16_initial.reshape(16, 16, 728)

I = I_real[:, start:ending]  # 2020.12.25
R = R_real[:, start:ending]  # 2021.02.23

# 用于预测控制
i2 = np.zeros((L, T + 7))
a2 = np.zeros((L, T + 7))
x2 = np.zeros((L, T + 7))
r2 = np.zeros((L, T + 7))
ir2 = np.zeros((L, T + 7))
i2[:, :7] = I[:, -7:]
a2[:, :7] = 0.86 * I[:, -7:]
x2[:, :7] = 0.14 * I[:, -7:]
r2[:, :7] = R[:, -7:]
ir2[:, :7] = I[:, -7:] + R[:, -7:]
i2 = i2.transpose()
a2 = a2.transpose()
x2 = x2.transpose()
r2 = r2.transpose()
ir2 = ir2.transpose()
# Real values are used for comparison
I_r = I_real[:, ending - 1:ending + T]  # 16*31 340-370
R_r = R_real[:, ending - 1:ending + T]

real_i = I_r.transpose()
real_r = R_r.transpose()
i_previous = [[0 for i in range(L)] for j in range(T)]
r_previous = [[0 for i in range(L)] for j in range(T)]
i_previous = I_r.transpose()[:-1]
r_previous = R_r.transpose()[:-1]

# Fit parameters for predicting the next 30 days for comparison
I_fp = np.zeros((L, T))
R_fp = np.zeros((L, T))
I_fpdata = pd.read_csv(r'.\I_fitted_pre2.csv')
I_fpdata_value = I_fpdata.values[:, :]  # 16*91
I_fp = I_fpdata_value
I_fitted_pre = I_fp


# sair real data
I2_r = I_real[:, ending - 1:ending + T]
A2_r = 0.86 * I_real[:, ending - 1:ending + T]
X2_r = 0.14 * I_real[:, ending - 1:ending + T]  # 16*31 340-370
R2_r = R_real[:, ending - 1:ending + T]
real_i2 = I2_r.transpose()
real_a2 = A2_r.transpose()
real_x2 = X2_r.transpose()
real_r2 = R2_r.transpose()
i2_previous = [[0 for i in range(L)] for j in range(T)]
r2_previous = [[0 for i in range(L)] for j in range(T)]
i2_previous = I2_r.transpose()
r2_previous = R2_r.transpose()

# fitted prediction
i2_fp_xiancun_data = pd.read_csv(r'.\1.6_SAIR_xiancun.csv')
i2_fp_xiancun_value = i2_fp_xiancun_data.values[:, :]  # 16*91
i2_fp_xiancun = i2_fp_xiancun_value
i2_fitted_pre_xiancun = i2_fp_xiancun

# SAIR参数 PRM拟合结果
belta_a = np.array(
    [0.17568501, 1.37493131, 0.11735176, 6.15997922, 0.01110436, 0.01616497, 0.02704505, 0.01691228, 0.00762972,
     0.28001221, 0.25259522, 0.11518299, 0.13753097, 0.03330608, 0.13483204, 0.28006249])
belta_x = np.array(
    [0.31942728728399326, 2.49987510565818063, 0.21336684401879187, 11.19996221711067522, 0.02018974547338731,
     0.02939085145865612, 0.04917281014259143, 0.03074960104052878, 0.0138722113254412, 0.50911311756037978,
     0.45926404092300717, 0.20942361534380042, 0.25005631306692435, 0.06055651358719044, 0.2451491575402375,
     0.50920451944502963])
epsilon = np.array(
    [0.04667894922824811, 0.04021405548913359, 0.04466735213160176, 0.042213429433096084, 0.04077805739857607,
     0.04046540388756729, 0.04162519088935855, 0.04092910430802354, 0.040748653521106584, 0.04372031074572422,
     0.038753272494514684, 0.04444668991183946, 0.047170990888174176, 0.03926429919919682, 0.0452921660458761,
     0.044015230006275184])
r_a = np.array(
    [0.04667894922824811, 0.04021405548913359, 0.09466735213160176, 0.042213429433096084, 0.04077805739857607,
     0.04046540388756729, 0.04162519088935855, 0.04092910430802354, 0.040748653521106584, 0.04372031074572422,
     0.038753272494514684, 0.04444668991183946, 0.047170990888174176, 0.03926429919919682, 0.0452921660458761,
     0.044015230006275184])
r_s = np.array(
    [0.07997483704947229, 0.06399398961876432, 0.080926051533376965, 0.63999816061484269, 0.089123249158105854,
     0.08933405836766409, 0.089679695837483854, 0.13937605116270621, 0.12865564427774152, 0.26595915343287806,
     0.189965596378785335, 0.43988270068203084, 0.139959030243992076, 0.239572816737005294, 0.239916710314392194,
     0.13996346232581881])
BA_all = np.diag(belta_a)
BX_all = np.diag(belta_x)

N = 5  # prediction horizon
D = 4  # Maximum triggering duration
Tmax = 31
p = 0.1
q = 0.2
global u_history_temp
u_history_temp_init1 = [0.5 for i in range(80)]
tmp_init = 0.55
daily_resource = -3
PSO_speed = 0.008
PSO_iter = 200
PSO_size = 50
u_history_temp = flatten([np.array(u_history_temp_init1)])
LARGE_INT_MIN = 0.05
LARGE_INT_MAX = 0.95
LARGE_INT = 1000000
save_csv_plot_path = f'.\\1.3result_PSO_max-3_init0.5 0.55_speed0.08_CNN_1616_without_sair_event_uaverage_200iter_50size\\'

ts = [0, 6]  # Two dimensional, only saving the date of the most recently triggered u used
all_days = [ts[0]]  # Stores the date for time/event updates without distinction

is_satisfied_condition = None  # 事件驱动标志位
is_satisfied_time = None  # 时间驱动标志位

init_count = 1
# Store the predicted values for 30 days
a2_history = [0 for i in range(T)]
x2_history = [0 for i in range(T)]
i2_history = [0 for i in range(T)]  # i2 = a2+x2
r2_history = [0 for i in range(T)]
ir2_history = [0 for i in range(T)]
u_history = [0 for i in range(T)]
u_history[-1] = [0 for i in range(L)]
ctrl_vars_history = [0 for i in range(T + 5)]
obj_fit_history = [0 for i in range(T)]
obj_g_history = [0 for i in range(T)]
obj_fit_history1 = [0 for i in range(T)]
obj_g_history1 = [0 for i in range(T)]
real_model_output_history = [0 for i in range(T - 1)]

for k in range(6, 6 + Tmax):  # k：6~35
    _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=f'☆☆☆   Calculating the {k+1-6} th day   ☆☆☆')
    with open('log.txt', 'a') as f:
        f.write(str(_))
        f.write('\n')
    print(f'☆☆☆   Calculating the {k+1-6} th day   ☆☆☆')
    if k == 6:
        i2[k] = real_i2[k - 6]  # i7 = real_i0
        a2[k] = real_a2[k - 6]
        x2[k] = real_x2[k - 6]
        r2[k] = real_r2[k - 6]
        is_satisfied_condition = True
    else:
        # step2: 更新系统结构
        '''
        u_opt = ctrl_vars_history[k - ts[1]]  # u[距离上次触发的天数]  6-6，
        E = np.eye(16)
        I = E * i[k]
        R = E * r[k]
        U = E * u_opt
        '''
        # u_opt = ctrl_vars_history[k - 6] ## k-7? ## 事件驱动咋整,解决：滑动更新，有更新就变，没更新就还是上一个策略的u2345里的，这个和casadi中的不一样，casadi是只执行u1
        # u_history[k-7] = ctrl_vars_history[ts[1] - 7]
        u_history[k - 7] = [(x+y+z+a+b)/5 for x,y,z,a,b in zip(ctrl_vars_history[ts[1] - 7], ctrl_vars_history[ts[1] - 6], ctrl_vars_history[ts[1] - 5], ctrl_vars_history[ts[1] - 4], ctrl_vars_history[ts[1] - 3])]
        real_model_output_history[k - 7] = model_runner(flatten_and_divide(np.array(ctrl_vars_history[ts[1] - 7]) * W_16[:, :, ending + 96 + k - 8]))
        obj_fit_history[k - 7] = obj_fit_history1[ts[1] - 7]  ## 记载每次的最优适应度值
        obj_g_history[k - 7] = obj_g_history1[ts[1] - 7]
        # u_opt = ctrl_vars_history[ts[1] - 7]
        u_opt = [(x+y+z+a+b)/5 for x,y,z,a,b in zip(ctrl_vars_history[ts[1] - 7], ctrl_vars_history[ts[1] - 6], ctrl_vars_history[ts[1] - 5], ctrl_vars_history[ts[1] - 4], ctrl_vars_history[ts[1] - 3])]
        E = np.eye(16)
        U1 = np.diag(u_opt)
        I2 = np.diag(i2[k - 1])
        A2 = np.diag(a2[k - 1])
        X2 = np.diag(x2[k - 1])
        R2 = np.diag(r2[k - 1])

        a2[k] = a2[k - 1] + 0.86 * ((E - A2 - X2 - R2) @ BA_all @ U1 @ W_16[:, :, ending + 96 + k - 7] @ a2[k - 1] + (
                E - A2 - X2 - R2) @ BX_all @ U1 @ W_16[:, :, ending + 96 + k - 7] @ x2[k - 1]) - np.diag(
            a2[k - 1]) @ epsilon - np.diag(a2[k - 1]) @ r_a
        x2[k] = x2[k - 1] + 0.14 * ((E - A2 - X2 - R2) @ BA_all @ U1 @ W_16[:, :, ending + 96 + k - 7] @ a2[k - 1] + (
                E - A2 - X2 - R2) @ BX_all @ U1 @ W_16[:, :, ending + 96 + k - 7] @ x2[k - 1]) + np.diag(
            a2[k - 1]) @ epsilon - np.diag(x2[k - 1]) @ r_s
        r2[k] = r2[k - 1] + np.diag(a2[k - 1]) @ r_a + np.diag(x2[k - 1]) @ r_s
        i2[k] = a2[k] + x2[k]  # 现存
        ir2[k] = a2[k] + x2[k] + r2[k]  # 累计

    i2_history[k - 6] = list(i2[k])  # i_history = i[7] ，存储数据
    a2_history[k - 6] = list(a2[k])
    x2_history[k - 6] = list(x2[k])  # i_history = i[7] ，存储数据
    r2_history[k - 6] = list(r2[k])
    ir2_history[k - 6] = list(ir2[k])

    # step3: Determine whether the triggering conditions have been met or the maximum triggering duration has been reached
    sum2 = np.sum(i2[k - 1] * population) / np.sum(population)
    if k < 21:  # k starting from 6, update condition 1.1-15 days 2.16-31 days
        event_t = 0.00095
    else:
        event_t = 0.00063
    if sum2 > event_t:
        is_satisfied_condition = True
    elif k - ts[-1] >= D:
        is_satisfied_time = True
    else:
        is_satisfied_condition = False
        is_satisfied_time = False
        continue

    # step4: Trigger response processing
    if is_satisfied_condition or is_satisfied_time:
        is_satisfied_condition = None
        is_satisfied_time = None

        ts[0] = ts[1]
        ts[1] = k + 1
        all_days.append(ts[1])

        # Variable preparation
        tmp = [tmp_init for i in range(80)]
        ctl_var1 = np.array(tmp[:16])
        ctl_var2 = np.array(tmp[16:32])
        ctl_var3 = np.array(tmp[32:48])
        ctl_var4 = np.array(tmp[48:64])
        ctl_var5 = np.array(tmp[64:80])
        ctl_vars = [ctl_var1, ctl_var2, ctl_var3, ctl_var4, ctl_var5]

        try:
            while True:
                # step4.1:prepare data model input and calculate data model output
                try:
                    compose2group = [0 for i in range(5)]
                    if len(ctl_vars) == 80:
                        ctl_vars = [
                            np.array(ctl_vars[:16]),
                            np.array(ctl_vars[16:32]),
                            np.array(ctl_vars[32:48]),
                            np.array(ctl_vars[48:64]),
                            np.array(ctl_vars[64:80])
                        ]

                    for uindex, uitem in enumerate(ctl_vars):
                        compose2group[uindex] = uitem

                    model_input = [0 for i in range(5)]
                    for group_index, group_item in enumerate(compose2group):
                        tmp = flatten_and_divide(group_item * W_16[:, :, ending + 96 + k - 7])
                        model_input[group_index] = tmp

                    model_input = np.array(model_input)
                except Exception as e:
                    _ = organize_log.generate_log_msg(success=True, msg_code='E0006', \
                                                      msg=f'Running error 0! Error message:{e}')
                    with open('log.txt', 'a') as f:
                        f.write(str(_))
                        f.write('\n')

                try:
                    total_model_output = np.array([0])
                    for count in range(5):
                        model_output = model_runner(model_input[count])
                        total_model_output = total_model_output + model_output
                except Exception as e:
                    _ = organize_log.generate_log_msg(success=True, msg_code='E0006', \
                                                      msg=f'Running error 1! Error message:{e}')
                    with open('log.txt', 'a') as f:
                        f.write(str(_))
                        f.write('\n')

                # step4.2 Calculate the g-constraint value and determine whether u1~u5 comply with the g-constraint
                g = total_model_output - daily_resource * (D + 1)

                # step4.3：以i(k+1)，r(k+1)为系统初值，依次输入u1~u5，计算新的i1~i5,r1~r5
                opt_i2 = np.array([list(i2[k])] + [[0 for i in range(16)] for j in range(5)])
                opt_a2 = np.array([list(a2[k])] + [[0 for i in range(16)] for j in range(5)])
                opt_x2 = np.array([list(x2[k])] + [[0 for i in range(16)] for j in range(5)])
                opt_r2 = np.array([list(r2[k])] + [[0 for i in range(16)] for j in range(5)])
                for index in range(5):
                    E = np.eye(16)
                    U = E * ctl_vars[index]

                    I3 = E * opt_i2[index]
                    A3 = E * opt_a2[index]
                    X3 = E * opt_x2[index]
                    R3 = E * opt_r2[index]

                    opt_a2[index + 1] = opt_a2[index] + 0.86 * (
                                (E - A3 - X3 - R3) @ BA_all @ W_16[:, :, ending + 96 + k - 7] @ np.array(
                            opt_a2[index]).T + (E - A3 - X3 - R3) @ BX_all @ W_16[:, :, ending + 96 + k - 7] @ np.array(
                            opt_x2[index])).T - np.diag(opt_a2[index]) @ epsilon - np.diag(opt_a2[index]) @ r_a
                    opt_x2[index + 1] = opt_x2[index] + 0.14 * (
                                (E - A3 - X3 - R3) @ BA_all @ W_16[:, :, ending + 96 + k - 7] @ opt_a2[index] + (
                                    E - A3 - X3 - R3) @ BX_all @ W_16[:, :, ending + 96 + k - 7] @ opt_x2[
                                    index]) + np.diag(opt_a2[index]) @ epsilon - np.diag(opt_x2[index]) @ r_s
                    opt_r2[index + 1] = opt_r2[index] + np.diag(opt_a2[index]) @ r_a + np.diag(opt_x2[index]) @ r_s
                    opt_i2[index + 1] = opt_a2[index + 1] + opt_x2[index + 1] + opt_r2[index + 1]

                # PSO initialization
                if init_count == 1:
                    Algorithm_Finished = False
                    PSO_Generator = PSO(
                        [LARGE_INT_MIN for i in range(80)],
                        [LARGE_INT_MAX for i in range(80)],
                        80,
                        PSO_iter,
                        PSO_size,
                        PSO_speed,
                        0.6,
                        2,
                        2,
                        flatten(ctl_vars),
                        daily_resource=daily_resource,
                        D=D
                    )
                    ctl_vars = next(PSO_Generator)
                    init_count = 0

                patient = cost_func(opt_a2[1:], opt_x2[1:], p, q)
                f = patient

                ctl_vars = PSO_Generator.send(f)
                if len(ctl_vars) == 3:
                    is_ready = True
                    best_var, best_fit, g = ctl_vars[0], ctl_vars[1], ctl_vars[2]
                    Algorithm_Finished = True
                    _ = organize_log.generate_log_msg(success=True, msg_code='I0000', \
                                                      msg=f"#######################################################" + \
                                                          f"Optimal control variable：{best_var} \n" + f"Optimal fitness: {best_fit}" + \
                                                          f"#######################################################")
                    with open('log.txt', 'a') as f:
                        f.write(str(_))
                        f.write('\n')

                    ctrl_vars_history[k - 6] = list(best_var[:16])
                    ctrl_vars_history[k - 5] = list(best_var[16:32])
                    ctrl_vars_history[k - 4] = list(best_var[32:48])
                    ctrl_vars_history[k - 3] = list(best_var[48:64])
                    ctrl_vars_history[k - 2] = list(best_var[64:80])

                    u_history_temp1 = np.array(best_var[:16])
                    u_history_temp2 = np.array(best_var[16:32])
                    u_history_temp3 = np.array(best_var[32:48])
                    u_history_temp4 = np.array(best_var[48:64])
                    u_history_temp5 = np.array(best_var[64:80])
                    u_history_temp = [u_history_temp1, u_history_temp2, u_history_temp3, u_history_temp4,
                                      u_history_temp5]
                    u_history_temp = flatten(u_history_temp)

                    obj_fit_history1[k - 6] = best_fit
                    obj_g_history1[k - 6] = g

                if Algorithm_Finished == True:
                    init_count = 1
                    round_best_var = []
                    round_best_result = []
                    Algorithm_Finished = None
                    best_var, best_fit = None, None
                    break
        except Exception as e:
            _ = organize_log.generate_log_msg(success=True, msg_code='E0006', \
                                              msg=f'Running error! Error message:：{e}')
            with open('log.txt', 'a') as f:
                f.write(str(_))
                f.write('\n')

# result
i2_afterwards = i2_history
ir2_afterwards = ir2_history
T_COUNT = 31

print(real_model_output_history)
# np.savetxt('PSO_i_previous.csv', i_previous, delimiter=',')
np.savetxt(save_csv_plot_path + f'6_PSO_data_model_i2_afterwards_max{daily_resource}.csv', i2_afterwards, delimiter=',')
np.savetxt(save_csv_plot_path + f'6_PSO_data_model_ir2_afterwards_max{daily_resource}.csv', ir2_afterwards,
           delimiter=',')
np.savetxt(save_csv_plot_path + f'6_PSO_data_model_real_output_max{daily_resource}.csv', real_model_output_history,
           delimiter=',')
for dimension in range(16):
    date = [str(day + 1) for day in range(T_COUNT)]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    plot_x = range(len(date))
    plt.tight_layout()
    plt.plot(date, [i2_previous[day][dimension] for day in range(T_COUNT)], color='blueviolet', marker='x',
             linestyle='-', label='real data')
    plt.plot(date, [i2_afterwards[day][dimension] for day in range(T_COUNT)], color='blue', marker='|', linestyle='--',
             label='SAIR PSO result')
    plt.plot(date, [i2_fitted_pre_xiancun[day][dimension] for day in range(T_COUNT)], color='red', marker='x',
             linestyle='--', label='SAIR prediction')
    plt.legend(loc='lower left')

    plt.xticks(plot_x, date, rotation=75, fontsize=8)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title('Infection number node ' + str(dimension))
    plt.savefig(save_csv_plot_path + f'dimension{dimension + 1}（PSO_daily_resource{daily_resource}）.png', dpi=400)
    plt.clf()

date = [str(day + 1) for day in range(T_COUNT - 1)]
plt.rcParams['font.sans-serif'] = ['SimHei']
plot_x = range(len(date))
plt.tight_layout()
# plot
plt.plot(date, [daily_resource for day in range(T_COUNT - 1)], color='blueviolet', marker='o', linestyle='-',
         label='economic index')
plt.plot(date, [real_model_output_history[day] for day in range(T_COUNT - 1)], color='blue', marker='o', linestyle='-',
         label='real output')
# plt.plot(date, [sum(I_fitted_pre[day]) for day in range(T_COUNT)], color='red', marker='o', linestyle='-', label='prediction without control')
plt.title('economic index_' + str(daily_resource))
plt.legend(loc='lower left')
plt.xticks(plot_x, date, rotation=75, fontsize=8)
plt.xlabel("time")
plt.ylabel("value")
plt.savefig(save_csv_plot_path + f'economic index_{daily_resource}.png', dpi=400)
plt.clf()

# Store optimization variable iteration records and corresponding objective function values
csv_file_path = f'6_PSO_data_model_opt_results_max{daily_resource}.csv'
csv_file_path1 = f'6_PSO_data_model_opt_results_event_max{daily_resource}.csv'
for i in range(T_COUNT):
    for j in range(16):
        row = [str(i + 1)] + [j for j in ctrl_vars_history[i]] + [obj_fit_history[i]] + [obj_g_history[i]] + [j for j in u_history[i]]
        row1 = [str(i + 1)] + [j for j in u_history[i]]

    try:
        with open(save_csv_plot_path + csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        with open(save_csv_plot_path + csv_file_path1, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row1)
    except Exception as e:
        _ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=e)
all_days1 = all_days
for i in range(1, len(all_days)):
    all_days1[i] = all_days[i] - 6
_ = organize_log.generate_log_msg(success=True, msg_code='I0000', msg=f'Trigger day record：{all_days}')
print(f'Trigger day record：{all_days}')
with open('log.txt', 'a') as f:
    f.write(str(_))
    f.write('\n')
_ = organize_log.generate_log_msg(success=True, msg_code='I0000',
                                  msg=f'real_model_output_history: {real_model_output_history}')
with open('log.txt', 'a') as f:
    f.write(str(_))
    f.write('\n')