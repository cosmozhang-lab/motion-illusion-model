# Neuron base

import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support import CommonConfig
from program import chain2

T_EXCITATORY = 1
T_INHIBITORY = 2
T_EXC = T_EXCITATORY
T_E = T_EXCITATORY
T_INH = T_INHIBITORY
T_I = T_INHIBITORY

T_ON = 3
T_OFF = 4
T_O = T_ON
T_F = T_OFF

class NeuronGroup:
  """
  config:
      coor: a tuple that contains (x,y,z) coordinates, each as a np.array
  """
  def __init__(self, nshape, config = CommonConfig()):
    self.nshape = nshape
    self.nneurons = int(np.prod(nshape)) # Number of neurons
    if config.coor:
      coor = config.coor
      if len(coor) > 0: self._x = clspt.Variable( np.array(coor[0]).astype(np.double), read_only = True )
      if len(coor) > 1: self._y = clspt.Variable( np.array(coor[1]).astype(np.double), read_only = True )
      if len(coor) > 2: self._z = clspt.Variable( np.array(coor[2]).astype(np.double), read_only = True )
    
    """
    Recording spikes.
    Each neuron will spike once at most in each time bin, the probability
    of spiking is firing_rate*dt. So each group will spike nneurons times
    at most. So we use nneurons-sized buffers to record the spikes.
    Remarks: As we need to inspect the spikes step-by-step, we cannot
    accelerate this operation with OpenCL, so we do not need Variable here.
    """
    # number of spikes in the time bin
    self.nspikes = 0 
    # Time of spikes
    # In iteration, this is used to record the time of spike of each neuron.
    # In this case the index is the neuron index.
    # After iteration, we rearrange this so that the spikes are arranged in
    # time sequence. In this case the index is the spike index.
    self.tspikes = np.zeros((self.nneurons,)).astype(np.double)
    # Which neuron spiked
    self.ispikes = np.zeros((self.nneurons,)).astype(np.int32)


  def step(self, t, dt):
    pass
