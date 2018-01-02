import numpy as np
from largescale.src.support.geometry import gen_coordinates

def make_gabor(size=(0,0), orientation=0, scale=0, period=0, phase=0, peak=1.0):
  if isinstance(scale, tuple):
    xscale = scale[1]
    yscale = scale[0]
  else:
    xscale = scale
    yscale = scale
  [coorx, coory] = gen_coordinates(size, center_zero=True)
  cos_o = np.cos(orientation / 180.0 * np.pi)
  sin_o = np.sin(orientation / 180.0 * np.pi)
  # radius = np.sqrt(coorx**2 + coory**2)
  gabor = np.exp( - ((coorx/xscale)**2 + (coory/yscale)**2) / 2 )
  yy = coorx * sin_o + coory * cos_o
  yyy = 2 * np.pi * yy / period - phase
  return peak * np.cos(yyy) * gabor
