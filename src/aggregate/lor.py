import numpy as np
import math


class LORSelector:

    def __init__(self, sampling_zero=0.05):
        self.sampling_zero = 0.05


    # @log_odd_ratio
    def odd_ratio(self, A: int, B: int, C: int, D: int, N: int):
        A = float(A) + self.sampling_zero
        B = float(B) + self.sampling_zero
        C = float(C) + self.sampling_zero
        D = float(D) + self.sampling_zero
        N = A + B + C + D
        deter = A * (N - B)
        non_deter = B * (N - A)
        return math.log((deter / non_deter), math.e)

    def fit(self, data, target):
        # print(data)
        # print(target)
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
            self.term_scores[c] = [c, self.odd_ratio(A, B, C, D, N)]

        return

    def scored_features(self, features):
        # print(data)
        # print(target)
        features_scores = []
        target_indices = []
        for temp_score in self.term_scores:
            features_scores.append([features[temp_score[0]], temp_score[1]])

        features_scores = sorted(features_scores, key=lambda term: term[1], reverse=True)
        return features_scores

    # @fit_odd_ratio
    def transform(self, data, negation=False):
        # print(data)
        # print(target)
        temp_scores = []
        target_indices = []
        for temp_score in self.term_scores:
            if negation:
                temp_scores.append([temp_score[0], -1 * temp_score[1]])
            else:
                temp_scores.append([temp_score[0], temp_score[1]])

        temp_scores = sorted(temp_scores, key= lambda term: term[1], reverse=True)

        for x in range(len(temp_scores)):
            if temp_scores[x][1] >= 0.0:
                print(temp_scores[x][0], temp_scores[x][1])
                target_indices.append(temp_scores[x][0])

        print(len(data))
        print(data.shape)
        print(list(frozenset(target_indices)))
        return data[: , list(frozenset(target_indices))]


