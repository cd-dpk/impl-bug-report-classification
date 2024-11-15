import numpy as np
import math

class FeatureSelector:

    def __init__(self, sampling_zero=0.05):
        self.sampling_zero = 0.05
        self.pos_fs = []
        self.neg_fs = []

    def odd_ratio(self, A: int, B: int, C: int, D: int, N: int):
        A = float(A) + self.sampling_zero
        B = float(B) + self.sampling_zero
        C = float(C) + self.sampling_zero
        D = float(D) + self.sampling_zero
        N = float(N) + 2 * self.sampling_zero
        deter = A * (N - B)
        non_deter = B * (N - A)
        return math.log((deter / non_deter))

    def signed_info_gain(self, A: int, B: int, C: int, D: int, N: int):
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

            return (A*D-B*C) * ig

    def fit_transform_odd_ratio(self, data, target, l: int, l1_ratio: float):
        # print(data)
        # print(target)
        self.pos_fs = []
        self.neg_fs = []
        term_scores = np.zeros(len(data[0]), dtype=list)
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
            print(A,B,C,D,N)
            term_scores[c] = [c, self.odd_ratio(A,B,C,D,N)]

        # print(term_scores)

        pos_term_scores = []
        neg_term_scores = []
        for x in term_scores:
            pos_term_scores.append([x[0],  x[1]])
            neg_term_scores.append([x[0],  x[1]])

        print(pos_term_scores)
        print(neg_term_scores)

        pos_term_scores = sorted(pos_term_scores, key=lambda term: term[1], reverse=True)
        neg_term_scores = sorted(neg_term_scores, key=lambda term: term[1])

        print(pos_term_scores)
        print(neg_term_scores)

        l1 = int(l * l1_ratio)
        for x in range(l1):
            self.pos_fs.append(pos_term_scores[x][0])

        l2 = int(l-l1)
        for x in range(l2):
            self.neg_fs.append(neg_term_scores[x][0])

        # print(self.pos_fs)
        # print(self.neg_fs)

        data_trf = data[:,self.pos_fs+self.neg_fs]
        return data_trf

    def get_lexicon_terms(self, data, target):
        term_scores = np.zeros(len(data[0]), dtype=list)
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
            print(A,B,C,D,N)
            term_scores[c] = [c, self.odd_ratio(A,B,C,D,N)]
            print(term_scores[c])

        # print(term_scores)

        pos_term_scores = []
        neg_term_scores = []
        neu_term_scores = []

        for term_score in term_scores:
            if term_score[1] > 0.0:
                pos_term_scores.append([term_score[0], term_score[1]])
            elif term_score[1] < 0.0:
                neg_term_scores.append([term_score[0], -1.0*term_score[1]])
            else:
                neu_term_scores.append([term_score[0], term_score[1]])

        print('Pos', len(pos_term_scores))
        print('Neg', len(neg_term_scores))
        print('Neu', len(neu_term_scores))

        return (pos_term_scores, neg_term_scores, neu_term_scores)


    def get_pos_neg_feature_terms(self, data, target):
        self.pos_fs = []
        self.neg_fs = []
        term_scores = np.zeros(len(data[0]), dtype=list)
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
            print(A,B,C,D,N)
            term_scores[c] = [c, self.odd_ratio(A,B,C,D,N)]
            print(term_scores[c])

        # print(term_scores)

        pos_term_scores = []
        neg_term_scores = []
        for x in term_scores:
            pos_term_scores.append([x[0],  x[1]])
            neg_term_scores.append([x[0],  -1*x[1]])

        print(pos_term_scores)
        print(neg_term_scores)

        pos_term_scores = sorted(pos_term_scores, key=lambda term: term[1], reverse=True)
        neg_term_scores = sorted(neg_term_scores, key=lambda term: term[1], reverse=True)

        print(pos_term_scores)
        print(neg_term_scores)

        return (pos_term_scores, neg_term_scores)



    def transform_odd_ratio(self,data):
        return data[:,self.pos_fs+self.neg_fs]

    def fit_transform_info_gain(self, data, target, l: int, l1_ratio:float):
        print(data)
        print(target)
        self.pos_fs = []
        self.neg_fs = []
        term_scores = np.zeros(len(data[0]), dtype=list)
        t_Cp = np.zeros([len(data[0]), 4], dtype=int)
        Cp = 1
        print(term_scores.shape)
        print(t_Cp.shape)
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
            N = len(t_Cp)
            print(A,B,C,D,N)
            term_scores[c] = [c, self.signed_info_gain(A,B,C,D,N)]

        print(term_scores)

        pos_term_scores = []
        neg_term_scores = []
        for x in term_scores:
            pos_term_scores.append([x[0],  x[1]])
            neg_term_scores.append([x[0],  x[1]])

        print(pos_term_scores)
        print(neg_term_scores)

        pos_term_scores = sorted(pos_term_scores, key=lambda term: term[1], reverse=True)
        neg_term_scores = sorted(neg_term_scores, key=lambda term: term[1])

        print(pos_term_scores)
        print(neg_term_scores)

        l1 = int(l * l1_ratio)
        for x in range(l1):
            self.pos_fs.append(pos_term_scores[x][0])

        l2 = int(l-l1)
        for x in range(l2):
            self.neg_fs.append(neg_term_scores[x][0])

        print(self.pos_fs)
        print(self.neg_fs)

        data_trf = data[:,self.pos_fs+self.neg_fs]
        return data_trf

    def transform_info_gain(self, data):
        return data[:,self.pos_fs+self.neg_fs]
