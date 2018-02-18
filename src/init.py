import numpy as np
import math


def odd_ratio(A:int,B:int,C:int,D:int,N:int):
    A = float(A)
    B = float(B)
    C = float(C)
    D = float(D)
    N = float(N)
    if A == 0 or B == 0 or C == 0 or D == 0:
        return 0.0
    else:
        deter = A * (N - B)
        print(deter)
        non_deter = B * (N - A)
        print(non_deter)
        print(deter / non_deter)

        return (deter / non_deter)

def info_gain(A:int,B:int,C:int,D:int,N:int):
    A = float(A)
    B = float(B)
    C = float(C)
    D = float(D)
    N = float(N)

    if A == 0.0 or B == 0.0 or C == 0.0 or D == 0.0:
        return 0.0
    else:
        ig = 0.0

        ig += A / N
        ig *= math.log((A * N) / ((A + B) * (A + C)), math.e)

        ig += (C / N)
        ig += math.log((C * N) / ((C + D) * (A + C)), math.e)

        ig += B / N
        ig *= math.log((B * N) / ((A + B) * (B + D)), math.e)

        ig += (D / N)
        ig *= math.log((D * N) / ((C + D) * (B + D)), math.e)

        return ig


def oddRation(t_C:np, N):
    odds = np.zeros(len(t_C), dtype=float)
    for c in range(len(t_C)):
        odds[c] = odd_ratio(t_C[c][0],t_C[c][1],t_C[c][2],t_C[c][3],N)
    return odds

def infoGain(t_C:np, N):
    igs = np.zeros(len(t_C),dtype=float)
    for c in range(len(t_C)):
        A = float(t_C[c][0])
        B = float(t_C[c][1])
        C = float(t_C[c][2])
        D = float(t_C[c][3])
        igs[c] = info_gain(A,B,C,D,N)
    return igs

data =[[1, 0, 0],[0,1,0],[0,0,0]]
target = [0,1,0]

print(data)
print(target)

t_Cp = np.zeros([len(data[0]), 4], dtype=int)
t_Cn = np.zeros([len(data[0]), 4], dtype=int)
Cp = 1
Cn = 0
for row in range(len(data)):
    for column in range(len(data[row])):
        if data[row][column] != 0 and target[row] == Cp:
            t_Cp[column][0] += 1
        elif data[row][column] != 0 and target[row] != Cp:
            t_Cp[column][1] += 1
        elif data[row][column] == 0 and target[row] == Cp:
            t_Cp[column][2] += 1
        elif data[row][column] == 0 and target[row] != Cp:
            t_Cp[column][3] += 1

        if data[row][column] != 0 and target[row] == Cn:
            t_Cn[column][0] += 1
        elif data[row][column] != 0 and target[row] != Cn:
            t_Cn[column][1] += 1
        elif data[row][column] == 0 and target[row] == Cn:
            t_Cn[column][2] += 1
        elif data[row][column] == 0 and target[row] != Cn:
            t_Cn[column][3] += 1

print("TERM_CP")
print(t_Cp)
print("TERM_CN")
print(t_Cn)

D = float(len(data))
print(D)

print(oddRation(t_Cp,D))
print(oddRation(t_Cn,D))