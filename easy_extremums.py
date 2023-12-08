import numpy as np
from scipy.optimize import fsolve

def equations(vars):
    x, y = vars
    eq1 = y / 5 - np.cos(x - 2 * y) - np.sin(x + 3 * y)
    eq2 = x / 5 + 2 * np.cos(x - 2 * y) - 3 * np.sin(x + 3 * y)
    return [eq1, eq2]

def vectorMagnitude(vector):
    res = np.sqrt(vector[0]**2 + vector[1]**2)
    if(res == 0):
        print("---------------")
        print(vector)
    return np.sqrt(vector[0]**2 + vector[1]**2)

def grid(xInterval, yInterval, xPointsAmount, yPointsAmount):
    xArray = np.linspace(xInterval[0], xInterval[1], xPointsAmount)
    yArray = np.linspace(yInterval[0], yInterval[1], yPointsAmount)
    result = []
    for x in xArray:
        for y in yArray:
            result.append(np.array([x, y]))
    return result

def isNewPoint(pointsList, newPoint):
    eps = 1e-3
    for point in pointsList:
        if vectorMagnitude(newPoint - point) <= 2 * eps:
            return False
    return True

def isInRange(xRange, yRange, Point):
    return xRange[0] < Point[0] and xRange[1] > Point[0] and yRange[0] < Point[1] and yRange[1] > Point[1]

"""
grid = grid([-5, 5], [-2, 2], 10, 10)
sols = []
for point in grid:
    solution = fsolve(equations, point)
    if isNewPoint(sols, solution) and isInRange([-5, 5], [-2, 2], point):
        sols.append(solution)

for solution in sols:
    print("Critical point coordinates (x, y):", solution)
print(len(sols))
"""
# Solving the system of equations numerically

import numpy as np

# Define the two sets of points
set1 = np.array([[-1.75442672,  1.61239042],
[-4.15810014,  0.38417222],
[-0.21729578, -0.94659235],
[2.18727754, 0.28136395],
[4.59027239, 1.50843892]])

set2 = np.array([
    [-5.01859272, -3.39901922], [-4.15810009, 0.38417225], [-120.05022069, 114.74720907],
    [-5.71689304, 2.95037703], [-2.62042406, -2.17395951], [-8.11464602, 1.720372],
    [-120.05846626, 114.75482101], [-1.06461399, -4.73609386], [-3.32110485, 4.18095308],
    [-0.21729577, -0.94659238], [1.33074948, -3.50868461], [-6.56007466, -0.84285925],
    [-1.75442702, 1.61239039], [-0.93087119, 5.41069673], [2.18727754, 0.28136393],
    [3.73094097, -2.27979391], [0.64788223, 2.84032336], [2.96000417, -6.09656177],
    [4.59027231, 1.50843891], [3.04541676, 4.06633338]
])

# Find points unique to set1
unique_to_set1 = []
for point1 in set1:
    found_similar = False
    for point2 in set2:
        if np.allclose(point1, point2, atol=1e-2):
            found_similar = True
            break
    if not found_similar:
        unique_to_set1.append(point1)

# Find points unique to set2
unique_to_set2 = []
for point2 in set2:
    found_similar = False
    for point1 in set1:
        if np.allclose(point1, point2, atol=1e-2):
            found_similar = True
            break
    if not found_similar:
        unique_to_set2.append(point2)

# Display the unique points in both sets
print("Points unique to set 1:")
print(unique_to_set1)
print(len(set1))
print(len(unique_to_set1))

print("\nPoints unique to set 2:")
print(unique_to_set2)
print(len(set2))
print(len(unique_to_set2))


"""
import numpy as np
from scipy.optimize import minimize

def f(x):
    y = x[0]
    y = x[1]
    return y * x[0] / 5 - np.sin(x[0] - 2 * x[1]) + np.cos(x[0] + 3 * x[1])

def vectorMagnitude(vector):
    res = np.sqrt(vector[0]**2 + vector[1]**2)
    if(res == 0):
        print("---------------")
        print(vector)
    return np.sqrt(vector[0]**2 + vector[1]**2)

def isInRange(xRange, yRange, Point):
    return xRange[0] < Point[0] and xRange[1] > Point[0] and yRange[0] < Point[1] and yRange[1] > Point[1]

def grid(xInterval, yInterval, xPointsAmount, yPointsAmount):
    xArray = np.linspace(xInterval[0], xInterval[1], xPointsAmount)
    yArray = np.linspace(yInterval[0], yInterval[1], yPointsAmount)
    result = []
    for x in xArray:
        for y in yArray:
            result.append(np.array([x, y]))
    return result

def isNewPoint(pointsList, newPoint):
    eps = 1e-3
    for point in pointsList:
        if vectorMagnitude(newPoint - point) <= 2 * eps:
            return False
    return True

# Bounds for x and y
bounds = [(-5, 5), (-2, 2)]

min_points = []  # To store the coordinates of minimum points

# Discretize the domain and find minimum points using minimize
for x in np.linspace(-5, 5, 100):
    for y in np.linspace(-2, 2, 100):
        result = minimize(f, (x, y), bounds=bounds)
        if result.success and isNewPoint(min_points, result.x):  # Check if optimization was successful
            min_points.append(result.x)

# Remove duplicate minimum points
min_points = np.unique(min_points, axis=0)

print("Number of minimum points found:", len(min_points))
print(min_points)
print("Coordinates of minimum points:")
for point in min_points:
    print(f"x: {point[0]}, y: {point[1]}")
"""
