import numpy as np
from numpy import array
import Funkcija as F
import gen_algorithm_functions as gaf
from my_utils import *
import math as m
from mayavi import mlab
import random as rng


xf, yf = np.mgrid[-4:4:50j, -4:4:50j]
z = F.function(xf, yf)
fun_plot = prepare_plot(xf, yf, z)
gen_size = 20
xd = np.random.rand(gen_size)*6-3
yd = np.random.rand(gen_size)*6-3


@mlab.animate(delay=3000)
def update_animation(xd, yd):
    zd = F.function(xd, yd)
    s = mlab.points3d(xd, yd, zd, color=(0, 0, 0), scale_factor=0.5)
    for iii in range(1, 1000):
        x_y_enc = gaf.encode(xd, yd, -3, 3, -3, 3, 10**-3)
        fitness = gaf.fitness_function(xd, yd)
        print(f"Iteracija: {iii}")
        for x, y, x_ys, fit in zip(xd, yd, x_y_enc, fitness):
            print(f"\t\t x:{x}\t\t y:{y}\t\t encoded:{np.flip(x_ys)}\t\t fitness:{fit}\t\t")
        x_y_fitness = [(x, fit) for x, fit in zip(x_y_enc, fitness)]
        selection = gaf.tournament_selection(x_y_fitness, 2, 6)
        new_gen = []
        for i in range(0, m.floor(len(selection)/2)):
            new_gen.extend(gaf.recombine(selection[2*i][0], selection[2*i+1][0], 1))
        x_y_fitness.sort(key=lambda xx: xx[1], reverse=True)
        selection.sort(key=lambda xx: xx[1], reverse=True)
        while not len(new_gen) == gen_size:
            if not len(selection) == 0:
                new_gen.append(selection.pop(0)[0])
            else:
                new_gen.append(x_y_fitness.pop(0)[0])
        new_gen = [gaf.mutation(x, 0.3) for x in new_gen]
        x_y_enc = new_gen
        xd, yd = gaf.decode(x_y_enc, -3, 3, -3, 3, 10**-3)
        zd = F.function(xd, yd)
        s.mlab_source.set(x=xd, y=yd, z=zd)
        
        yield


a = update_animation(xd, yd)
mlab.show()

