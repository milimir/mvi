import math as m
import numpy as np
def function(x, y):
    """x: niz koordinata na x osi
       y: niz koordinata po y osi
       vraca niz koordinata po z osi
       predstavlja funkciju za koju trazimo minimum """
    prvi_dio = 3*(1-x)**2*np.exp(-x**2-(y+1)**2)
    drugi_dio = 10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)
    treci_dio = 1/3*np.exp(-(x+1)**2-y**2)
    return prvi_dio-drugi_dio-treci_dio
