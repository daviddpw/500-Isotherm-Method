import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import *


def conc_comp(e_comp, fc):
    if fc < 50 * 10 ** 6:
        e_c2 = 0.002
        e_cu2 = 0.0035
        n = 2
    else:
        e_c2 = (2 + 0.085 * (fc / 10 ** 6 - 50) ** 0.53) / 1000
        e_cu2 = (2.6 + 35 * ((90 - fc / 10 ** 6) / 100) ** 4) / 1000
        n = 1.4 + 23.4 * ((90 - fc / 10 ** 6) / 100) ** 4

    if 0 <= e_comp < e_c2:       # e_comp is the strain level of concrete in compression
        stress = fc * (1 - (1 - e_comp / e_c2) ** n)
    elif e_c2 <= e_comp <= e_cu2:
        stress = fc
    else:
        stress = 0

    return stress


def conc_ten(Ec, fct, fR1, fR3, l_NA, e_ten):
    w_u = 2.5 * 10 ** (-3)
    e_ct = fct / Ec
    e_ftu = e_ct + w_u / l_NA     # l_NA is the location of neutral axis

    if 0 <= e_ten <= e_ct:
        stress = Ec * e_ten
    elif e_ten <= e_ftu:
        stress = fR1 - (fR1 - fR3) * (e_ten - e_ct) / (e_ftu - e_ct)
    else:
        stress = 0

    return stress


def steel_fire(fsy, Es, e_steel, T_n):
    # all the units are in Pa
    fsp = 1 * fsy

    #reduction of fsy, fsp, Es
    if 0 <= T_n <= 100:
        T_fsy = 1
        T_fsp = 1
        T_Es = 1
    elif 100 < T_n <= 200:
        T_fsy = 1
        T_fsp = 1 - 0.19 / 100 * (T_n - 100)
        T_Es = 1 - 0.1 / 100 * (T_n - 100)
    elif 200 < T_n <= 300:
        T_fsy = 1
        T_fsp = 0.81 - 0.2 / 100 * (T_n - 200)
        T_Es = 0.9 - 0.1 / 100 * (T_n - 200)
    elif 300 < T_n <= 400:
        T_fsy = 1
        T_fsp = 0.61 - 0.19 / 100 * (T_n - 300)
        T_Es = 0.8 - 0.1 / 100 * (T_n - 300)
    elif 400 < T_n <= 500:
        T_fsy = 1 - 0.22 / 100 * (T_n - 400)
        T_fsp = 0.42 - 0.06 / 100 * (T_n - 400)
        T_Es = 0.7 - 0.1 / 100 * (T_n - 400)
    elif 500 < T_n <= 600:
        T_fsy = 0.78 - 0.31 / 100 * (T_n - 500)
        T_fsp = 0.36 - 0.18 / 100 * (T_n - 500)
        T_Es = 0.6 - 0.29 / 100 * (T_n - 500)
    elif 600 < T_n <= 700:
        T_fsy = 0.47 - 0.24 / 100 * (T_n - 600)
        T_fsp = 0.18 - 0.11 / 100 * (T_n - 600)
        T_Es = 0.31 - 0.18 / 100 * (T_n - 600)
    elif 700 < T_n <= 800:
        T_fsy = 0.23 - 0.12 / 100 * (T_n - 700)
        T_fsp = 0.07 - 0.02 / 100 * (T_n - 700)
        T_Es = 0.13 - 0.04 / 100 * (T_n - 700)
    elif 800 < T_n <= 900:
        T_fsy = 0.11 - 0.05 / 100 * (T_n - 800)
        T_fsp = 0.05 - 0.01 / 100 * (T_n - 800)
        T_Es = 0.09 - 0.02 / 100 * (T_n - 800)
    elif 900 < T_n <= 1000:
        T_fsy = 0.06 - 0.02 / 100 * (T_n - 900)
        T_fsp = 0.04 - 0.02 / 100 * (T_n - 900)
        T_Es = 0.07 - 0.03 / 100 * (T_n - 900)
    elif 1000 < T_n <= 1100:
        T_fsy = 0.04 - 0.02 / 100 * (T_n - 1000)
        T_fsp = 0.02 - 0.01 / 100 * (T_n - 1000)
        T_Es = 0.04 - 0.02 / 100 * (T_n - 1000)
    else:
        T_fsy = 0
        T_fsp = 0
        T_Es = 0

    fsy_T = fsy * T_fsy
    fsp_T = fsp * T_fsp
    Es_T = Es * T_Es
    e_sp = fsp_T / Es_T
    e_sy = 0.02
    e_st = 0.15
    e_su = 0.20

    c_p = (fsy_T - fsp_T) ** 2 / ((e_sy - e_sp) * Es_T - 2 * (fsy_T - fsp_T))
    a_p = ((e_sy - e_sp) * (e_sy - e_sp + c_p / Es_T)) ** 0.5
    b_p = (c_p * (e_sy - e_sp) * Es_T + c_p ** 2) ** 0.5

    if 0 <= e_steel <=e_sp:
        stress = e_steel * Es * T_Es
    elif e_sp <= e_steel <= e_sy:
        stress = fsp_T - c_p + (b_p / a_p) * (a_p ** 2 - (e_sy - e_steel) ** 2) ** 0.5
    elif e_sy <= e_steel <= e_st:
        stress = fsy_T
    elif e_st <= e_steel <= e_su:
        stress = fsy_T * (1 - (e_steel - e_st) / (e_su - e_st))
    else:
        stress = 0

    return stress


def interaction_curve(dx, width, height, cover, fc, Ec, fys, Es, fts, D_sr, N_com_bar, N_ten_bar, fct, fR1, fR3, length, BC, reduction, T_n):
    width = width - 2 * reduction
    height = height - 2 * reduction
    cover = cover - reduction

    Nx = int(round(height / float(dx)))
    x = np.linspace(0, height, Nx + 1)
    dx = x[1] - x[0]

    ec_comp = np.zeros(Nx + 1)
    stress_cc = np.zeros(Nx + 1)
    force_cc = np.zeros(Nx + 1)
    moment_cc = np.zeros(Nx + 1)
    total_force_cc = np.zeros(Nx + 1)
    total_moment_cc = np.zeros(Nx + 1)
    es_comp = np.zeros(Nx + 1)
    stress_sc = np.zeros(Nx + 1)
    force_sc = np.zeros(Nx + 1)
    moment_sc = np.zeros(Nx + 1)
    total_force_ct = np.zeros(Nx + 1)
    total_moment_ct = np.zeros(Nx + 1)
    es_ten = np.zeros(Nx + 1)
    stress_st = np.zeros(Nx + 1)
    force_st = np.zeros(Nx + 1)
    moment_st =np.zeros(Nx + 1)

    total_N = np.zeros(Nx + 2)
    total_M = np.zeros(Nx + 2)

    NB = np.zeros(Nx + 2)
    MMF = np.zeros(Nx + 2)
    force_column = np.zeros(Nx + 2)
    moment_column = np.zeros(Nx + 2)
    y_e1 = np.zeros(Nx + 2)
    y_e2 = np.zeros(Nx + 2)
    y_e3 = np.zeros(Nx + 2)

    if fc < 50 * 10 ** 6:
        e_c2 = 0.002
        e_cu2 = 0.0035
        n = 2
    else:
        e_c2 = (2 + 0.085 * (fc / 10 ** 6 - 50) ** 0.53) / 1000
        e_cu2 = (2.6 + 35 * ((90 - fc / 10 ** 6) / 100) ** 4) / 1000
        n = 1.4 + 23.4 * ((90 - fc / 10 ** 6) / 100) ** 4

    for i in range(1, Nx + 1):
        l_NA = i * dx     # l_NA is the location of Neutral axis
        rem = height - l_NA
        if rem == 0:
            rem = 0.005
        w_u = 2.5 * 10 ** (-3)
        e_com_lim = e_cu2
        e_ten_lim = fct / Ec + w_u / rem

        #determine failure type, i.e. compression failure or tension failure, both concrete; convert back to com_lim
        if (e_com_lim / l_NA) > (e_ten_lim / rem):
            e_lim = e_ten_lim / rem * l_NA
        else:
            e_lim = e_com_lim

        #concrete in compression
        for j in range (0, i + 1):
            ec_comp[j] = j * dx / l_NA * e_lim          # concrete strain in compression
            stress_cc[j] = conc_comp(ec_comp[j], fc)    # stress of concrete in compression
            force_cc[j] = stress_cc[j] * width * dx     # force of concrete in compression

            if l_NA <= height / 2:
                moment_cc[j] = force_cc[j] * (height / 2 - (i - j) * dx)     # moment of concrete in compression
            else:
                neg_c = int(round((l_NA - height / 2) / dx))
                if j < neg_c:
                    moment_cc[j] = -1 * force_cc[j] * (l_NA - height / 2 - j * dx)
                else:
                    moment_cc[j] = force_cc[j] * (height / 2 - (i - j) * dx)

            total_force_cc[i] += force_cc[j]
            total_moment_cc[i] += moment_cc[j]      # total force and moment of concrete in comp

        #steel in compression
        if cover < 0:
            es_comp[i] = (l_NA - cover) / l_NA * e_lim
            sign_sc = 1
        else:
            if l_NA >= cover:
                sign_sc = 1
                es_comp[i] = e_lim * (l_NA - cover) / l_NA
            else:
                sign_sc = -1
                es_comp[i] = e_lim * (cover - l_NA) / l_NA

        stress_sc[i] = steel_fire(fys, Es, es_comp[i], T_n)

        force_sc[i] = stress_sc[i] * (math.pi * D_sr ** 2 / 4) * N_com_bar * sign_sc
        moment_sc[i] = force_sc[i] * (0.5 * height - cover)

        #concrete in tension
        m = int(round(rem / dx))
        ec_ten = np.zeros(m + 1)
        stress_ct = np.zeros(m + 1)
        force_ct = np.zeros(m + 1)
        moment_ct = np.zeros(m + 1)

        for j in range (0, m + 1):
            ec_ten[j] = j * dx / l_NA * e_lim  # concrete strain in compression
            stress_ct[j] = conc_ten(Ec, fct, fR1, fR3, rem, ec_ten[j])
            force_ct[j] = -1 * stress_ct[j] * width * dx       # concrete is in tension; negative

            if l_NA >= (height / 2):
                moment_ct[j] = -1 * force_ct[j] * (l_NA - height / 2 + j * dx)
            else:
                neg_t = int(round((height / 2 - l_NA) / dx))
                if j <= neg_t:
                    moment_ct[j] = force_ct[j] * (height / 2 - l_NA - j * dx)
                else:
                    moment_ct[j] = -1 * force_ct[j] * (j * dx - (height / 2 - l_NA))

            total_force_ct[i] += force_ct[j]
            total_moment_ct[i] += moment_ct[j]

        #steel in tension
        if cover < 0:
            es_ten[i] = (height - cover - l_NA) / l_NA * e_lim
            sign_st = -1
        else:
            if l_NA <= (height - cover):
                sign_st = -1
                es_ten[i] = e_lim * (height - cover - l_NA) / l_NA
            else:
                sign_st = 1
                es_ten[i] = e_lim * (l_NA - height + cover) / l_NA

        stress_st[i] = steel_fire(fys, Es, es_ten[i], T_n)

        force_st[i] = stress_st[i] * (math.pi * D_sr ** 2 / 4) * N_ten_bar * sign_st     #steel is in tension; negative
        moment_st[i] = force_st[i] * (0.5 * height - cover) * -1

        total_N[i] = (total_force_cc[i] + force_sc[i] + total_force_ct[i] + force_st[i]) / 1000
        total_M[i] = (total_moment_cc[i] + moment_sc[i] + total_moment_ct[i] + moment_st[i]) / 1000

    #pure tension
    total_N[0] = -(width * height * fR3 + (N_com_bar + N_ten_bar) * (math.pi * D_sr ** 2 / 4) * fys) / 1000
    total_M[0] = 0

    #pure compression
    total_N[Nx + 1] = (width * height * fc * 0.85 + (N_com_bar + N_ten_bar) * (math.pi * D_sr ** 2 / 4) * fys) / 1000
    total_M[Nx + 1] = 0

    #N-M interaction curve of cross-section
    # print('force')
    # for i in range(0, Nx + 2):
    #     print(total_N[i])
    #
    # print('moment')
    # for i in range(0, Nx + 2):
    #     print(total_M[i])

    Is = ((D_sr / 2) ** 4 * math.pi / 4 + math.pi * D_sr ** 2 / 4 * (height / 2 - cover) ** 2) * 4
    Ic = width * height ** 3 / 12

    k1 = (fc / 10 ** 6 / 20) ** 0.5
    lamda = 12 ** 0.5 * length / width
    Ac = width * height
    for i in range (0, Nx + 2):
        k2 = total_N[i] * 1000 / (Ac * fc) * lamda / 170
        if k2 > 0.2:
            k2 = 0.2
        EI = k1 * k2 * Ec * Ic + Es * Is
        NB[i] = math.pi ** 2 * EI / (length * BC) ** 2
        if total_N[i] >=0:
            MMF[i] = 1 / (1 - total_N[i] * 1000 / NB[i])  # moment magnification factor
        else:
            MMF[i] = 1
        moment_column[i] = total_M[i] / MMF[i]   # column moment
        force_column[i] = total_N[i]

    #N-M interaction curve of column
    # print('moment')
    # for i in range(0, Nx + 2):
    #     print(moment_column[i])

    # print('force')
    # for i in range(0, Nx + 2):
    #     print(force_column[i])

    e1 = (height + 2 * reduction) / 3 * 100
    # e1 = (height) / 3 * 100
    # e1 = 10
    e2 = 30
    e3 = 50

    #find intersection point
    for i in range(0, Nx + 1):
        y_e1[i] = moment_column[i] * 1000 / e1
        y_e2[i] = moment_column[i] * 1000 / e2
        y_e3[i] = moment_column[i] * 1000 / e3

    idx_1 = np.argwhere(np.diff(np.sign(y_e1 - force_column)) != 0).reshape(-1) + 0  # intersection with column capacity
    idx_2 = np.argwhere(np.diff(np.sign(y_e2 - force_column)) != 0).reshape(-1) + 0
    idx_3 = np.argwhere(np.diff(np.sign(y_e3 - force_column)) != 0).reshape(-1) + 0

    y_0 = 1000 / e1 * force_column[Nx + 1] / (1000 / e1 + (force_column[Nx + 1] - force_column[Nx]) / moment_column[Nx])
    # y_0 = force_column[Nx + 1] * moment_column[Nx] * e3 / (moment_column[Nx] * e3 + force_column[Nx + 1] - force_column[Nx])

    print(y_0)
    print(y_e1[idx_1+1])
    print(y_e2[idx_2+1])
    print(y_e3[idx_3+1])

    #plot
    plt.plot(total_M, total_N, 'b-')
    plt.legend(['N-M'], loc='upper right')
    plt.xlabel('M')
    plt.ylabel('N')
    xmax = total_M.max() * 1.2
    ymax = total_N.max() * 1.2
    ymin = total_N.min() * 1.2
    plt.axis([0, xmax, ymin, ymax])

    # show()

dx = 0.001
width = 0.27
height = 0.27
cover = 50 * 10 ** (-3)    #cover, to center of rebar
fc = 93 * 10 ** 6
Ec = 40 * 10 ** 9
fys = 565 * 10 ** 6
Es = 200 * 10 ** 9
fts = 565 * 10 ** 6
D_sr = 25 * 10 ** (-3)
N_com_bar = 2
N_ten_bar = 2
fct = 11.7 * 10 ** 6      # Pa
fR1 = 10.14 * 10 ** 6
fR3 = 8.2 * 10 ** 6       # Pa
length = 3.14
BC = 1
reduction = 0 * 10 ** (-3)
T_n = 30

interaction_curve(dx, width, height, cover, fc, Ec, fys, Es, fts, D_sr, N_com_bar, N_ten_bar, fct, fR1, fR3, length, BC, reduction, T_n)