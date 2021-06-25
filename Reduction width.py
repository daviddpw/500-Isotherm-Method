import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def solver(M, dt, dx, p_a, B, h_0, s, e_0):
    global k_T
    global p_T
    global c_T
    Nt = int(round(M / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)
    Nx = int(round(B / float(dx)))
    x = np.linspace(0, B, Nx + 1)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # colorinterpolation = 50
    # colourMap = plt.cm.jet

    #creat a 2D array
    T = np.zeros(shape = (Nx+1, Nx+1))
    T_n = np.zeros(shape = (Nx+1, Nx+1))
    k_n = np.zeros(shape = (Nx+1, Nx+1))
    c_n = np.zeros(shape = (Nx+1, Nx+1))
    p_n = np.zeros(shape = (Nx+1, Nx+1))
    F_n = np.zeros(shape = (Nx+1, Nx+1))

    for i in range(0, Nx+1):         #intial condition
        for j in range(0, Nx+1):
            T_n[i][j] = 30

    for n in range(0, Nt+1):           #compute Temp and set boundary condition
        for i in range(0, Nx+1):
            for j in range(0, Nx+1):

                if 20 <= T_n[i][j] <= 1200:
                    # k_n[i][j] = (1.36 - 0.136 * (T_n[i][j] / 100) + 0.0057 * (T_n[i][j] / 100) ** 2)     #NSC
                    k_n[i][j] = (2 - 0.2451 * (T_n[i][j] / 100) + 0.0107 * (T_n[i][j] / 100) ** 2)         #HSC
                else:
                    k_n[i][j] = 0

                if 20 <= T_n[i][j] <= 100:
                    c_n[i][j] = 900
                elif 100 < T_n[i][j] <= 115:
                    c_n[i][j] = 2770
                elif 115 < T_n[i][j] <= 200:
                    c_n[i][j] = 2770 - 1770 * (T_n[i][j] - 115) / 85
                elif 200 < T_n[i][j] <= 400:
                    c_n[i][j] = 1000 + (T_n[i][j] - 200) / 2
                elif 400 < T_n[i][j] <= 1200:
                    c_n[i][j] = 1100
                else:
                    c_n[i][j] = 1100

                if 20 <= T_n[i][j] <= 115:
                    p_n[i][j] = p_a
                elif 115 < T_n[i][j] <= 200:
                    p_n[i][j] = p_a * (1 - 0.02 * (T_n[i][j] - 115) / 85)
                elif 200 < T_n[i][j] <= 400:
                    p_n[i][j] = p_a * (0.98 - 0.03 * (T_n[i][j] - 200) / 200)
                elif 400 < T_n[i][j] <= 1200:
                    p_n[i][j] = p_a * (0.95 - 0.07 * (T_n[i][j] - 400) / 800)
                else:
                    p_n[i][j] = 2150

        # Tf_n = 20 + 345 * np.log10(8 * n * dt / 60 + 1)    #ISO834

        #fire curve in this study
        if n < 1200:
            Tf_n = 30 + np.log10(1 * n / 60 + 1) * 450
        elif n < 3300:
            Tf_n = 625 + 10.5 * (n / 60 - 20)
        else:
            Tf_n = 992.5 + 5 * (n / 60 - 55)

        Ta_n = 30

        T_n[0][0] = Tf_n
        T_n[0][Nx] = Tf_n
        T_n[Nx][0] = Tf_n
        T_n[Nx][Nx] = Tf_n

        for i in range(1, Nx):
            F_n[i][0] = dt / (2 * p_n[i][0] * c_n[i][0] * dx ** 2)
            F_n[i][Nx] = dt / (2 * p_n[i][Nx] * c_n[i][Nx] * dx ** 2)

            T[i][0] = T_n[i][0] + F_n[i][0] * ((3*k_n[i][0]+k_n[i][1])*(T_n[i][1]-T_n[i][0]) +
                                    4*dx*h_0*(Tf_n-T_n[i][0]) + 4*dx*s*e_0*((Tf_n+273.15)**4 - (T_n[i][0]+273.15)**4))
            T[i][Nx] = T_n[i][Nx] + F_n[i][Nx] * ((3*k_n[i][Nx]+k_n[i][Nx-1])*(T_n[i][Nx-1]-T_n[i][Nx]) +
                                    4*dx*h_0*(Tf_n-T_n[i][Nx]) + 4*dx*s*e_0*((Tf_n+273.15)**4 - (T_n[i][Nx]+273.15)**4))

        for j in range(1, Nx):
            F_n[0][j] = dt / (2 * p_n[0][j] * c_n[0][j] * dx ** 2)
            F_n[Nx][j] = dt / (2 * p_n[Nx][j] * c_n[Nx][j] * dx ** 2)
            T[0][j] = T_n[0][j] + F_n[0][j] * ((3*k_n[0][j]+k_n[1][j])*(T_n[1][j]-T_n[0][j]) +
                                    4*dx*h_0*(Tf_n-T_n[0][j]) + 4*dx*s*e_0*((Tf_n+273.15)**4 - (T_n[0][j]+273.15)**4))
            T[Nx][j] = T_n[Nx][j] + F_n[Nx][j] * ((3*k_n[Nx][j]+k_n[Nx-1][j])*(T_n[Nx-1][j]-T_n[Nx][j]) +
                                    4*dx*h_0*(Tf_n-T_n[Nx][j]) + 4*dx*s*e_0*((Tf_n+273.15)**4 - (T_n[Nx][j]+273.15)**4))

        for i in range(1, Nx):
            for j in range(1, Nx):
                F_n[i][j] = dt/(2*p_n[i][j]*c_n[i][j]*dx**2)
                T[i][j] = T_n[i][j] + F_n[i][j]*((k_n[i+1][j]+k_n[i][j])*(T_n[i+1][j]-T_n[i][j]) - (k_n[i][j]+k_n[i-1][j])*(T_n[i][j]-T_n[i-1][j]) +\
                                                 (k_n[i][j+1]+k_n[i][j])*(T_n[i][j+1]-T_n[i][j]) - (k_n[i][j]+k_n[i][j-1])*(T_n[i][j]-T_n[i][j-1]))
        interval = 300  # display interval
        upp_lim = int(round(M / interval)) + 1
        for p in range(0, upp_lim):
            if n == interval * p:
                    print('time')
                    print(p * 5)
                    print("Rebar")
                    print(T_n[20][25])
                    # loc = int(round(B / 2 / float(dx))) + 1
                    # temp_list1 = np.zeros(loc + 1)
                    # temp_list2 = np.zeros(loc + 1)
                    # cen = int(round(B / 2 / float(dx)))
                    temp_list_1 = np.zeros(Nx + 1)
                    temp_list_2 = np.zeros(Nx + 1)

                    for i in range(0, Nx + 1):
                        temp_list_1[i] = T[i][67]  # temperature of center line in the new list
                        temp_list_2[i] = T[i][i]
                    print(temp_list_1)
                    print(temp_list_2)
                    temp_loc_1 = next(temp_list_1[0] for temp_list_1 in enumerate(temp_list_1) if
                                    temp_list_1[1] < 499)  # find the first point to be below centain value
                    temp_loc_2 = next(temp_list_2[0] for temp_list_2 in enumerate(temp_list_2) if
                                    temp_list_2[1] < 499)  # find the first point to be below centain value
                    print("reduction of width")
                    print(temp_loc_1)
                    print(temp_loc_2)

        T_n, T = T, T_n             #switch variables before next step

    # print("Final")
    # temp_list = np.zeros(Nx+1)        #a new list to contain the temperature
    # for i in range(0, Nx+1):
    #     temp_list[i] = T[i][50]        #temperature of center line in the new list
    # print(temp_list)
    # temp_loc = next(temp_list[0] for temp_list in enumerate(temp_list) if temp_list[1] < 499)   #find the first point to be below centain value
    # print(temp_loc)

    # plt.title("Contour of Temperature")
    # plt.contourf(x, x, T_n, colorinterpolation, cmap=colourMap)
    # plt.colorbar()
    # plt.show()

    return T_n, x, t

M = 8400
dt = 1
dx = 0.002
p_a = 2480
B = 0.27
h_0 = 25
s = 5.67*10**-8
e_0 = 0.7

solver(M, dt, dx, p_a, B, h_0, s, e_0)