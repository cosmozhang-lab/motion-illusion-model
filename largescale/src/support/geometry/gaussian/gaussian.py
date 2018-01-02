import numpy as np
from largescale.src.support.geometry import gen_coordinates

def make_gaussian(size=(0,0), scale=0, peak=1.0):
  if isinstance(scale, tuple):
    xscale = scale[1]
    yscale = scale[0]
  else:
    xscale = scale
    yscale = scale
  coors = gen_coordinates(size, center_zero=True)
  coory = coors[0].astype(np.float32)
  coorx = coors[1].astype(np.float32)
  return peak * np.exp( - ((coorx/xscale)**2 + (coory/yscale)**2) / 2 )
