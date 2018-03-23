from scipy.spatial import ConvexHull
class ROCAnalysis:

    def __init__(self,c_Y_n, c_N_p, p_n,p_p):
        self.c_Y_n = c_Y_n
        self.c_N_p = c_N_p
        self.p_n = p_n
        self.p_p = p_p
        self.m = (c_Y_n*p_n)/(c_N_p*p_p)

    def get_optimal_classifier_indices_using_convex_hull(self, points):
        hull = ConvexHull(points)
        print(len(hull.vertices))
        optimal_indices = []
        max_tp_intercept = -100
        for vertex in hull.vertices:
            tp_intercept = points[vertex,0] - self.m* points[vertex,1]
            if tp_intercept > max_tp_intercept:
                max_tp_intercept = tp_intercept
                optimal_indices.append(vertex)

        return optimal_indices
    def get_expected_cost(self,FP,TP):
        print(FP,TP)
        return self.p_p*(1-TP)*self.c_N_p+self.p_n*FP*self.c_Y_n