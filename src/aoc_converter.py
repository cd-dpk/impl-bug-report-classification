import numpy as np
import math, re
Security = 'Security'
Performance = 'Performance'
Subject = 'wicket'
i = 0
ratio = 0.5
file_name = '/media/geet/Random/PYTHON/simulated_data/'+Subject+'_'+Security+'_'+str(False)+'_'+str(i)+'_com_txt_fs_'+str(ratio)+'_log.txt'
in_file = open(file_name,'r')
points = [[0.0,0.0]]
counter = 0
for line in in_file:
    counter += 1
    if counter%4 == 0:
        tokens = re.split("[,\n]",line)
        if len(tokens) == 3:
            # print(line)
            points.append([float(str(tokens[0])), float(str(tokens[1]))])
points.append([1.0, 1.0])
print(file_name)
print(len(points))
print(points)
points = sorted(points, key=lambda  point: point[0])
print(points)
aoc = 0.0
for x in range(len(points)-1):
    # print(points[x+1][1],points[x][1],points[x+1][0],points[x][0],0.5)
    aoc += (points[x+1][1]+points[x][1])*abs(points[x+1][0]-points[x][0])*0.5
print(aoc)
