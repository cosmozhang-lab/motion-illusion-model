import numpy as np

def gen_coordinates(size, center_zero=False):
  ndims = len(size)
  retcoor = []
  for d in xrange(ndims):
    coorl = np.arange(size[d]).astype(np.float32)
    if center_zero: coorl = coorl - size[d] / 2
    coor = coorl
    for dd in xrange(d):
      coor = np.stack([coor for i in xrange(size[d-dd-1])], 0)
    for dd in xrange(d+1, ndims):
      coor = np.stack([coor for i in xrange(size[dd])], coor.ndim)
    retcoor.append(coor)
  return tuple(retcoor)
