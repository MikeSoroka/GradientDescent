import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math

def length(x,y):
    return np.sqrt(x**2 + y**2)


fig = plt.figure()
ex = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')

def surface(X,Y,Z):

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0,alpha = 0.6)
    #Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()



"""value for f(x)"""
def f(point):
    x = point[0]
    y = point[1]
    return y * x / 5 - np.sin(x - 2 * y) + np.cos(x + 3 * y)
    #return  -(200*x-3*x**2-2*x*y+150*y-2*y**2)

def fdx(point):
    dx = 1e-4
    x = point[0]
    y = point[1]
    return (f([x + dx, y]) - f([x - dx, y])) / (2 * dx)

def fdy(point):
    dy = 1e-4
    x = point[0]
    y = point[1]
    return (f([x, y + dy]) - f([x, y - dy])) / (2 * dy)

def fGradient(point):
    res = np.array([fdx(point), fdy(point)])
    #if math.isnan(res[0]):
        #print("---------------------")
        #print(point)
    return np.array([fdx(point), fdy(point)])

def vectorMagnitude(vector):
    res = np.sqrt(vector[0]**2 + vector[1]**2)
    if(res == 0):
        print("---------------")
        print(vector)
    return np.sqrt(vector[0]**2 + vector[1]**2)

def isInRange(xRange, yRange, Point):
    return xRange[0] <= Point[0] and Point[0] <= xRange[1] and yRange[0] <= Point[1] and Point[1] <= yRange[1]

def grid(xInterval, yInterval, xPointsAmount, yPointsAmount):
    xArray = np.linspace(xInterval[0], xInterval[1], xPointsAmount)
    yArray = np.linspace(yInterval[0], yInterval[1], yPointsAmount)
    result = []
    for x in xArray:
        for y in yArray:
            result.append(np.array([x, y]))
    return result

def isNewPoint(pointsList, newPoint):
    eps = 1e-6
    for point in pointsList:
        if vectorMagnitude(newPoint - point) <= 2 * eps:
            return False
    return True


def fNearestMinimum(stepModifier, startingPoint, iterationsLimit):
    eps = 1e-6
    realGradient = fGradient(startingPoint)
    step = 2
    currentGradient = realGradient / vectorMagnitude(realGradient)
    previousPoint = startingPoint
    currentPoint = startingPoint - currentGradient * step
    previousFValue = f(startingPoint)
    currentFValue = f(currentPoint)
    currentIteration = 1

    while vectorMagnitude(currentPoint - previousPoint) > eps and currentIteration < iterationsLimit:
        currentIteration += 1
        realGradient = fGradient(currentPoint)
        if abs(realGradient[0]) < eps and abs(realGradient[1]) < eps:
            break
        # normalizing a gradient
        currentGradient = realGradient / vectorMagnitude(realGradient)
        if currentFValue > previousFValue:
            step *= stepModifier
        previousPoint = currentPoint
        currentPoint = currentPoint - currentGradient * step
        previousFValue = currentFValue
        currentFValue = f(currentPoint)
        #print("Iteration:", currentIteration, "currentPoint:", currentPoint, "currentGradient", currentGradient, "currentStep:", step, "currentFValue", currentFValue)


    return currentPoint

def fNearestMaximum(stepModifier, startingPoint, iterationsLimit):
    eps = 1e-6
    realGradient = fGradient(startingPoint)
    step = 2
    currentGradient = realGradient / vectorMagnitude(realGradient)
    previousPoint = startingPoint
    currentPoint = startingPoint + currentGradient * step
    previousFValue = f(startingPoint)
    currentFValue = f(currentPoint)
    currentIteration = 1

    while vectorMagnitude(currentPoint - previousPoint) > eps and currentIteration < iterationsLimit:
        currentIteration += 1
        realGradient = fGradient(currentPoint)
        if abs(realGradient[0]) < eps and abs(realGradient[1]) < eps:
            break
        # normalizing a gradient
        currentGradient = realGradient / vectorMagnitude(realGradient)
        if currentFValue < previousFValue:
            step *= stepModifier
        previousPoint = currentPoint
        currentPoint = currentPoint + currentGradient * step
        previousFValue = currentFValue
        currentFValue = f(currentPoint)
        # print("Iteration:", currentIteration, "currentPoint:", currentPoint, "currentGradient", currentGradient, "currentStep:", step, "currentFValue", currentFValue)

    return currentPoint

gridData = open("grid.txt", "w")
extremumsData = open("extremums.txt", "w")
extremumsData.write("        Point        |       Minimum       |        Maximum      \n")
filteredMins = open("mins.txt", "w")
filteredMins.write("        Point        |       Minimum       \n")
filteredMaxs = open("maxs.txt", "w")
filteredMaxs.write("        Point        |       Maximum       \n")

minimumsList = []
maximumsList = []
grid = grid([-5, 5], [-2, 2], 6, 6)
#grid = [np.array([1, 1])]
for point in grid:
    gridData.write("Current point coordinates: (" + str(point[0]) + ";" + str(point[1]) +")\n")
    minimum = fNearestMinimum(0.3, point, 500)
    maximum = fNearestMaximum(0.3, point, 500)
    extremumsData.write("({0[0]:9.3f};{0[1]:9.3f})|({1[0]:9.3f};{1[1]:9.3f})|({2[0]:9.3f};{2[1]:9.3f})\n".format(
    point, minimum, maximum))
    #print(isNewPoint(extremumsList, extremum))
    if isNewPoint(minimumsList, minimum) and isInRange([-5, 5], [-2, 2], minimum):
        #print(isInRange([-5, 5], [-2, 2], point))
        #print(point)
        minimumsList.append(minimum)
    if isNewPoint(maximumsList, maximum) and isInRange([-5, 5], [-2, 2], maximum):
        #print(isInRange([-5, 5], [-2, 2], point))
        #print(point)
        maximumsList.append(maximum)

for minimum in minimumsList:
    filteredMins.write("{0:21.3f}|({1[0]:9.3f};{1[1]:9.3f})\n".format(
        f(minimum), minimum))
for maximum in maximumsList:
    filteredMaxs.write("{0:21.3f}|({1[0]:9.3f};{1[1]:9.3f})\n".format(
        f(maximum), maximum))
print(len(minimumsList))
print(len(maximumsList))


xx = np.arange(-5, 5, 0.1)
yy = np.arange(-2, 2, 0.1)
xx, yy = np.meshgrid(xx, yy) # meshgrid - a domain for plotting the function
zz = yy * xx / 5 - np.sin(xx - 2 * yy) + np.cos(xx + 3 * yy)


#print(grid)
surface(xx,yy,zz)
