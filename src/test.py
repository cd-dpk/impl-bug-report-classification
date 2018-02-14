from scipy.stats import fisher_exact

array =[[13, 32], [17, 23]]

n = array[0][0] + array[0][1] + array[1][0] + array[1][1]
nom = array[0][0] * (n - array[0][1])
denom = array[0][1] * (n - array[0][0])
odd = nom / denom
print(odd)
print(fisher_exact(array))