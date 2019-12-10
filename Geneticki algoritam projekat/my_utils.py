from mayavi import mlab
from mayavi.mlab import pipeline as pipe
import numpy as np


def prepare_plot(x, y, z):
    """funkcija priprema pozadintki plot tj f(x,y)
       sa svim oznakama"""
    fun_plot = pipe.array2d_source(x, y, z)
    fun_plot.add_module(mlab.axes(nb_labels=9, color=(0,0,0)))
    warp = pipe.warp_scalar(fun_plot)
    normals = pipe.poly_data_normals(warp)
    surf = pipe.surface(normals)
    mlab.scalarbar(orientation='vertical', title="f(x,y)")
    return fun_plot


def as_byte_bits(to_convert: np.ndarray, count: int) -> np.ndarray:
    """ to_convert: niz koji se pretvara u niz bitova(prestavljeni u memoriji bajtovima
        count: broj bita koje koristimo
        vraca niz bita"""
    to_ret = np.array([np.unpackbits(np.array([i]).view(np.uint8), bitorder='little', count=count) for i in to_convert])
    return to_ret


def from_byte_bits(to_convert: np.ndarray) -> np.ndarray:
    """to_convert: niz bita koje pretvaramo u integer
       vraca konvertovani niz"""
    if len(to_convert[0]) < 32:
        to_convert = np.array([np.append(i, [0 for o in range(0, 32-len(i))]) for i in to_convert])
    return np.array([np.packbits(i, bitorder='little').view(np.uint32)[0] for i in to_convert])

