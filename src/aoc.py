import numpy as np
import math, re
points =[]
points.append([0.0,0.0])
res_file = open('/media/geet/Random/PYTHON/simulated_data/wicket_Security_True_0_log.txt','r')
for line in res_file:
    tokens = re.split("[,\n]",line)
    if len(tokens) == 3:
        print(line)
        points.append([float(tokens[0]), float(tokens[1])])
points.append([1.0, 1.0])

aoc = 0.0
for x in range(len(points)-1):
    print(points[x+1][1],points[x][1],points[x+1][0],points[x][0],0.5)
    aoc += (points[x+1][1]+points[x][1])*abs(points[x+1][0]-points[x][0])*0.5
print(aoc)
