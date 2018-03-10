import numpy as np
import math

class MutualInformationSelector:

    def __init__(self):
        self.sampling_zero = 0.5

    def mutual_info(self, A:int, B:int, C:int, D:int, N:int):
        A = float(A) + self.sampling_zero
        B = float(B) + self.sampling_zero
        C = float(C) + self.sampling_zero
        D = float(D) + self.sampling_zero
        N = float(N) + 2 * self.sampling_zero

        mi = 0.0

        temp = (A / N) * math.log((A * N) / ((A + B) * (A + C)), math.e)
        mi += temp

        temp = (C / N) * math.log((C * N) / ((C + D) * (A + C)), math.e)
        mi += temp

        temp = (B / N) * math.log((B * N) / ((A + B) * (B + D)), math.e)
        mi += temp

        temp = (D / N) * math.log((D * N) / ((C + D) * (B + D)), math.e)
        mi += temp

        return mi


    # @fit
    def fit(self, data, target):
        self.term_scores = np.zeros(len(data[0]), dtype=list)
        t_Cp = np.zeros([len(data[0]), 4], dtype=int)
        Cp = 1
        # print(term_scores.shape)
        # print(t_Cp.shape)
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

        for c in range(len(t_Cp)):
            A = float(t_Cp[c][0])
            B = float(t_Cp[c][1])
            C = float(t_Cp[c][2])
            D = float(t_Cp[c][3])
            N = len(data)
            self.term_scores[c] = [c, self.mutual_info(A, B, C, D, N)]

        return

    # @transformed_data
    def transform(self, data, threshold):
        featured_indices = []
        for term_score in self.term_scores:
            if term_score[1] >= threshold:
                featured_indices.append(term_score[0])

        return data[:,featured_indices]

