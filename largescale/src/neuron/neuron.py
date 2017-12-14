# Neuron base

import numpy as np
import largescale.src.support.cl_support as clspt

class NeuronGroup:
  # config:
  #     coor: a tuple that contains (x,y,z) coordinates, each as a np.array
  def __init__(self, nshape, config = None):
    self.nshape = nshape
    self.nneurons = int(np.prod(nshape)) # Number of neurons
    if "coor" in config:
      coor = config["coor"]
      if len(coor) > 0: self._x = clspt.Variable( np.array(coor[0]).astype(np.double), read_only = True )
      if len(coor) > 1: self._y = clspt.Variable( np.array(coor[1]).astype(np.double), read_only = True )
      if len(coor) > 2: self._z = clspt.Variable( np.array(coor[2]).astype(np.double), read_only = True )

  def step(self, t, dt):
    pass
