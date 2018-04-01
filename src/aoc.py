import numpy as np
import math, re
points =[]
points.append([0.0, 0.0])
points.append([0.19290928690491582,0.6585365080309717])
points.append([1.0, 1.0])

aoc = 0.0
for x in range(len(points)-1):
    print(points[x+1][1],points[x][1],points[x+1][0],points[x][0],0.5)
    aoc += (points[x+1][1]+points[x][1])*abs(points[x+1][0]-points[x][0])*0.5
print(aoc)
