import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 1
y_0_1 = 0
z_0_1 = 0
y_0_2 = 0
z_0_2 = -1
y_n = np.exp(1) - 1


def function_1(x, y, z, delt_2):
    t = 1 / (np.exp(x) + 1)
    return t * z * (1 - delt_2) + (1 - t) * (y + 1)


def function_2(x, y, z, delt_2):
    t = 1 / (np.exp(x) + 1)
    return t * z * (1 - delt_2) + np.exp(x) * t * y


def function_solve(x):
    return np.exp(x) - 1


def method_1(y_0, z_0, function, epsilon, n_max, delt_2):
    def solve(section):
        step = (b - a) / section
        value_x_0 = a
        value_y_0 = y_0
        value_z_0 = z_0
        mas_x = [a]
        mas_y = [y_0]
        for j in range(section):
            approx_y = value_y_0 + 0.5 * step * value_z_0
            approx_z = value_z_0 + 0.5 * step * function(value_x_0, value_y_0, value_z_0, delt_2)
            value_y_1 = value_y_0 + 0.25 * step * (value_z_0 + approx_z)
            value_z_1 = value_z_0 + 0.25 * step * (
                    function(value_x_0, value_y_0, value_z_0, delt_2) + function(value_x_0 + 0.5 * step, approx_y,
                                                                                 approx_z, delt_2))
            approx_y = value_y_1 + 0.25 * step * (3 * value_z_1 - value_z_0)
            approx_z = value_z_1 + 0.25 * step * (
                    3 * function(value_x_0 + 0.5 * step, value_y_1, value_z_1, delt_2) - function(value_x_0, value_y_0,
                                                                                                  value_z_0, delt_2))
            value_y_0 = value_y_1
            value_z_0 = value_z_1
            value_y_1 = value_y_0 + 0.25 * step * (value_z_0 + approx_z)
            value_x_1 = value_x_0 + step
            value_z_1 = value_z_0 + 0.25 * step * (
                    function(value_x_0 + 0.5 * step, value_y_0, value_z_0, delt_2) + function(value_x_1, approx_y,
                                                                                              approx_z, delt_2))
            value_x_0 = value_x_1
            value_y_0 = value_y_1
            value_z_0 = value_z_1
            mas_x.insert(0, value_x_1)
            mas_y.insert(0, value_y_1)
        mas_x.reverse()
        mas_y.reverse()
        return mas_x, mas_y

    sec = 1
    error = 1
    result_solve = solve(sec)
    mas_error = []
    mas_section = [1]
    sec = 2
    while error > epsilon and sec < n_max:
        mas_y_1 = result_solve[1]
        result_solve = solve(sec)
        mas_y_2 = result_solve[1]
        error_summa = []
        for j in range(1, len(mas_y_1)):
            error_summa.insert(0, abs((mas_y_2[2 * j] - mas_y_1[j]) / 3))
        error = max(error_summa)
        mas_error.insert(0, error)
        mas_section.insert(0, sec)
        sec *= 2
    mas_x_final = result_solve[0]
    mas_y_final = result_solve[1]
    mas_error.reverse()
    mas_section.reverse()
    return mas_x_final, mas_y_final, mas_error, mas_section, error, sec / 2


def method_reduction(epsilon, n_max, delt_1, delt_2):
    res_1 = method_1(y_0_1, z_0_1, function_1, epsilon, n_max, delt_2)
    res_2 = method_1(y_0_2, z_0_2, function_2, epsilon, n_max, delt_2)
    length = len(res_2[1])
    c = (y_n * (1 - delt_1) - res_1[1][length - 1]) / res_2[1][length - 1]
    res_y = np.zeros((1, length))[0]
    for j in range(length):
        res_y[j] = res_2[1][j] * c + res_1[1][j]
    mas_error = []
    for j in range(len(res_1[2])):
        mas_error.insert(0, max(res_1[2][j], res_2[2][j]))
    mas_error.reverse()
    mas_sec = res_1[3]
    mas_sec.pop(0)
    error = max(res_1[4], res_2[4])
    return res_1[0], res_y, mas_error, mas_sec, error, res_1[5]


result_1_1 = method_reduction(1e-1, 1024, 0, 0)
result_1_2 = method_reduction(1e-3, 1024, 0, 0)
result_1_3 = method_reduction(1e-9, 4096, 0, 0)

plt.figure(1)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('График №1')
x_real = np.linspace(a, b, 1000)
y_real = function_solve(x_real)
plt.plot(x_real, y_real, label='real')
plt.plot(result_1_1[0], result_1_1[1], '-.', label='tol = 1e-1')
plt.plot(result_1_2[0], result_1_2[1], '--*', label='tol = 1e-3')
plt.legend()

plt.figure(2)
plt.grid()
plt.yscale('log')
plt.xlabel('section')
plt.ylabel('absolute error')
plt.title('График №2')
plt.plot(result_1_3[3], result_1_3[2], '--*', label='tol = 1e-9, n_max = 4096')
plt.legend()


# ------------------------------------------------build_graph_3_4-------------------------------------------------------

def graph_3_4():
    tol = 1e-1
    mas_tol = []
    mas_error = []
    mas_sec = []
    for j in range(10):
        mas_tol.insert(0, tol)
        result = method_reduction(tol, 10000, 0, 0)
        mas_error.insert(0, result[4])
        mas_sec.insert(0, result[5])
        tol = 0.1 * tol
    mas_tol.reverse()
    mas_error.reverse()
    mas_sec.reverse()
    return mas_tol, mas_error, mas_sec


check_2 = graph_3_4()

plt.figure(3)
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('tolerance')
plt.ylabel('error')
plt.title('График №3')
plt.plot(check_2[0], check_2[0], label='exact error')
plt.plot(check_2[0], check_2[1], label='experimental error')
plt.legend()

plt.figure(4)
plt.grid()
plt.xscale('log')
plt.xlabel('tolerance')
plt.ylabel('section')
plt.title('График №4')
plt.plot(check_2[0], check_2[2])
plt.legend()


# ------------------------------------------------build_graph_5_6-------------------------------------------------------

def graph_5(tol):
    delta = 0.0
    mas_delta = []
    mas_error = []
    mas_y_real = method_reduction(tol, 10000, 0, 0)[1]
    for j in range(5):
        mas_delta.insert(0, delta * 100)
        mas_y_approx = method_reduction(tol, 10000, delta, 0)[1]
        length = len(mas_y_approx)
        result = np.zeros((1, length))[0]
        for k in range(1, length):
            result[k] = abs((mas_y_real[k] - mas_y_approx[k]) / mas_y_real[k])
        mas_error.insert(0, max(result) * 100)
        delta += 0.1
    mas_delta.reverse()
    mas_error.reverse()
    return mas_delta, mas_error


check_7_1 = graph_5(1e-1)
check_7_2 = graph_5(1e-6)
#
plt.figure(5)
plt.suptitle('График №5')
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(check_7_2[0], check_7_2[1], label='tol = 1e-1')
plt.xlabel('start error, %')
plt.ylabel('relative error, %')
plt.legend()
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(check_7_1[0], check_7_1[1], 'r', label='tol = 1e-6')
plt.xlabel('start error, %')
plt.legend()


def graph_6(tol):
    delta_ = 0.0
    mas_delta = []
    mas_error = []
    mas_y_real = method_reduction(tol, 10000, 0, 0)[1]
    for j in range(5):
        mas_delta.insert(0, delta_ * 100)
        mas_y_approx = method_reduction(tol, 10000, 0, delta_)[1]
        length = len(mas_y_approx)
        result = np.zeros((1, length))[0]
        for k in range(1, length):
            result[k] = abs((mas_y_real[k] - mas_y_approx[k]) / mas_y_real[k])
        mas_error.insert(0, max(result) * 100)
        delta_ += 0.1
    return mas_delta, mas_error


check_8_1 = graph_6(1e-1)
check_8_2 = graph_6(1e-6)

plt.figure(6)
plt.suptitle('График №6')
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(check_8_2[0], check_8_2[1], label='tol = 1e-1')
plt.xlabel('error in equation, %')
plt.ylabel('relative error, %')
plt.legend()
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(check_8_1[0], check_8_1[1], 'r', label='tol = 1e-6')
plt.xlabel('error in equation, %')
plt.legend()

# ---------------------------------------------------graph_9------------------------------------------------------------


def graph_7():
    rel_error = []
    res = method_reduction(1e-6, 1024, 0, 0)
    error = res[2]
    length = len(error)
    for j in range(length):
        rel_error.insert(0, error[0] / error[j])
    rel_error.reverse()
    print(rel_error)
    return rel_error


plt.show()
