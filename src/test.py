import numpy as np
from builtins import print
from scipy.spatial import ConvexHull
sec_points = []   # 30 random points in 2-D
# sec_points = np.random.rand(30,2)
# (0.1898,0.8675)(0.1504,0.8466)(0.0858,0.7118)
#     (0.20089542,0.862248697)
#     (0.1522,0.7990)(0.2211,0.8369)(0.0970,0.7579)
sec_points.append([0.1898, 0.8675])
sec_points.append([0.1504, 0.8466])
sec_points.append([0.0858, 0.7118])
sec_points.append([0.20089542, 0.862248697])
sec_points.append([0.1522,0.7990])
sec_points.append([0.2211,0.8369])
sec_points.append([0.0970,0.7579])
sec_points = np.array(sec_points,dtype=float)
print(sec_points)
from src.jira.roc import ROCAnalysis
'''
roc_analysis = ROCAnalysis(2,15,6,1)
print(roc_analysis.get_expected_cost(sec_points[0][0],sec_points[0][1]))
print(roc_analysis.get_expected_cost(sec_points[3][0],sec_points[3][1]))
print(roc_analysis.get_expected_cost(sec_points[5][0],sec_points[5][1]))
# perf_points = np.random.rand(30,2)
# (0.2099,0.8437)(0.1473,0.8178)(0.0941,0.5351)
#  (0.184105805,0.831125828)
#    (0.16608,0.7934)(0.1056,0.7093)(0.0573,0.6523)
'''
perf_points = []
perf_points.append([0.2099,0.8437])
perf_points.append([0.1473,0.8178])
perf_points.append([0.0941,0.5351])
perf_points.append([0.184105805,0.831125828])
perf_points.append([0.16608,0.7934])
perf_points.append([0.1056,0.7093])
perf_points.append([0.0573,0.6523])
perf_points = np.array(perf_points,dtype=float)
roc_analysis = ROCAnalysis(1,5,6,1)
print(roc_analysis.get_expected_cost(perf_points[0][0],perf_points[0][1]))
print(roc_analysis.get_expected_cost(perf_points[3][0],perf_points[3][1]))
print(roc_analysis.get_expected_cost(perf_points[4][0],perf_points[4][1]))
