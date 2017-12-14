# Neuron base

import numpy as np
import largescale.src.support.cl_support as clspt

# class Neuron:
#   def __init__(self, coor=(0,0)):
#     self._coor = coor
#     self.x = self._coor[0] if len(self._coor) > 0 else 0
#     self.y = self._coor[1] if len(self._coor) > 1 else 0
#     self.z = self._coor[2] if len(self._coor) > 2 else 0
#     self.initialize()

#   def initialize(self):
#     self.last_spike_time = None

#   def step(self, t, dt):
#     pass

# class NeuronGroup:
#   def __init__(self, neurons=[]):
#     self.neurons = neurons

#   def at(idx):
#     if len(self.neurons) <= idx:
#       raise "Cannot get the neuron at %d: Index out of bounds." % idx
#     return self.neurons[idx]



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
