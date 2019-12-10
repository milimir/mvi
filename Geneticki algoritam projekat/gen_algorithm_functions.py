import numpy as np
import math as m
from my_utils import *
import Funkcija as F
import random as rng


def mutation(mutant_candidate: np.ndarray, mut_prob: float):
    """mutant_candidate: kandidat za mutaciju
       mut_prob: vjerovatnoca mutacije
       vrsi mutaciju gena tj odgovarajuci gen uveca za jedan i podjeli po modulu 2 da napravi logiku flipa kao kod bitskih operacija"""
    random = rng.random()

    if random < mut_prob:
        mut_loc = rng.randint(0, len(mutant_candidate)-1)
        np.put(mutant_candidate, [mut_loc], [(mutant_candidate[mut_loc]+1)%2])
    return mutant_candidate


def fitness_function(x: np.ndarray, y: np.ndarray)->list:
    """ x: niz x koordinata tacaka
        y:niz y koordinata tacaka
    prima dekodovane nizove koordinata x,y i vraca niz ocjena tacnosti
     za tacke koje obrazuju parovi iz x,y
     gdje najvece vrijednosti dobijaju losiju ocjenu a najmanje bolju"""
    zs = F.function(x, y)
    f_max = zs[np.argmax(zs)]
    return f_max-zs


def generation_eval(x: np.ndarray) -> float:
    """ x niz ocjena generacije
     vraca ukupnu ocjenu generacije(sumira)"""
    return x.sum()


def tournament_selection(x_ys_fitness: list, sel_size: int, k: int) -> list:
    """ x_ys_fitness: niz sa parovima (kodovane_koordinate,fitnes_za_koordinate)
        sel_size: broj parova za selekciju
        k: velicina turnira - broj slucajno izabranih jedinki iz x_ys od kojih se biraju 2 za rekombinaciju
    vraca kodovane tacke koje ce ucestvovati u rekombinaciji"""
    selection=[]
    for j in range(0, sel_size):
        if k > len(x_ys_fitness):
            k = len(x_ys_fitness)
        in_tournament = []
        for i in range(0, k):
            random_index = rng.randint(0, len(x_ys_fitness)-1)
            in_tournament.append(x_ys_fitness.pop(random_index))
        best = in_tournament[0]
        second_best = in_tournament[1]
        for to_check in in_tournament[:]:
            if to_check[1] > best[1]:
                second_best = best
                best = to_check

        in_tournament = [x for x in in_tournament if id(x) != id(best) and id(x) != id(second_best)]

        x_ys_fitness.extend(in_tournament)
        selection.append(best)
        selection.append(second_best)
    return selection


def recombine(x_y1: np.ndarray, x_y2: np.ndarray, no_rec_points: int) -> np.ndarray:
    """x_y1: prva jedinka za rekombinaciju
        x_y2: druga jedinka za rekombinaciju
       rec_prob: vjerovatnoca da ce doci do rekombinacije"""
    rec_point1 = m.floor(rng.random()*len(x_y1))
    rec_point2 = m.floor(rng.random()*len(x_y1)) if no_rec_points == 2 else len(x_y1)
    if rec_point1 > rec_point2:
        rec_point1, rec_point2 = rec_point2, rec_point1
    x_y1r, x_y2r = np.concatenate((x_y1[0:rec_point1], x_y2[rec_point1:rec_point2], x_y1[rec_point2:len(x_y1)])),\
        np.concatenate((x_y2[0:rec_point1], x_y1[rec_point1:rec_point2], x_y2[rec_point2:len(x_y2)]))
    return x_y1r, x_y2r


def encode(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, precision: float) -> list:
    """funkcija koduje x,y nizove brojeva u binarne nizove pogodnije za rad
       x,y: nizovi parova po koorinatama x,y
       x_min, x_max donja i gornja granica intervala po x osi respektivno
       y_min, y_max donja i gornja granica intervala po y osi respektivno
       precision okolina prihvatljive greske rjesenja
       vraca kodovan niz x i y u prihvaltljivom binarnom formatu i konkatenirane vrijednosti koordinata tacke po x,y"""
    x_count = m.ceil(m.log2((x_max-x_min)*(precision**-1)+1))
    y_count = m.ceil(m.log2((y_max-y_min)*(precision**-1)+1))
    x_bits = as_byte_bits(np.floor((x-x_min)/(x_max-x_min)*(2**x_count-1)).astype(np.int), x_count)
    y_bits = as_byte_bits(np.floor((y-y_min)/(y_max-y_min)*(2**y_count-1)).astype(np.int), y_count)
    return [np.append(i, j) for i, j in zip(x_bits, y_bits)]


def decode(x_y: list, x_min: float, x_max: float, y_min: float, y_max: float, precision: float):
    """"funkcija dekoduje x,y nizove brojeva iz binarnog niza za
    x_y: niz sa kodovanim koordinatama x,y
    x_min, x_max donja i gornja granica intervala po x osi respektivno
    y_min, y_max donja i gornja granica intervala po y osi respektivno
    precision okolina prihvatljive greske rjesenja
    vraca dekodovane nizove x i y u decimalnom formatu"""
    x_count = m.ceil(m.log2((x_max-x_min)*(precision**-1)+1))
    y_count = m.ceil(m.log2((y_max-y_min)*(precision**-1)+1))
    x_bits = np.array([i[0:x_count] for i in x_y])
    y_bits = np.array([i[x_count:x_count+y_count] for i in x_y])
    x = from_byte_bits(x_bits)*(x_max-x_min)/(2**len(x_bits[0])-1)+x_min
    y = from_byte_bits(y_bits)*(y_max-y_min)/(2**len(y_bits[0])-1)+y_min
    return x, y
