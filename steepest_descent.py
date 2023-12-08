def fSteepestDescent(stepModifier, startingPoint, iterationsLimit):
    eps = 1e-12
    realGradient = fGradient(startingPoint)
    if (abs(realGradient[0]) <= eps and abs(realGradient[1]) <= eps):
        return startingPoint
    step = 2
    currentGradient = realGradient / vectorMagnitude(realGradient)
    previousPoint = startingPoint
    currentPoint = startingPoint - currentGradient * step
    currentFValue = f(currentPoint)
    previousFValue = f(startingPoint)
    currentIteration = 1

    while (vectorMagnitude(currentPoint - previousPoint)
           and currentIteration < iterationsLimit):
        currentIteration += 1
        if currentFValue > previousFValue or currentIteration == 1:
            step *= stepModifier
            realGradient = fGradient(currentPoint)
            # normalizing a gradient
            currentGradient = realGradient / vectorMagnitude(realGradient)
        previousPoint = currentPoint
        currentPoint = currentPoint - currentGradient * step
        previousFValue = currentFValue
        currentFValue = f(currentPoint)
        #print("Iteration:", currentIteration, "currentPoint:", currentPoint)
        """
        if currentIteration > 2:
            plt.plot([currentPoint[0], previousPoint[0]], [currentPoint[1], previousPoint[1]],
                [currentFValue, previousFValue], c='r')
        """
    return currentPoint