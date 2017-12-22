# Neuron base

import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support.common import CommonConfig
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
    types: neuron types
    v_reset: reset voltage after spike
    v_thre: spike voltage threshold
    t_ref: refactory time
  """
  def __init__(self, nshape, config = CommonConfig()):
    self.nshape = nshape
    self.nneurons = int(np.prod(nshape)) # Number of neurons
    if config.coor:
      coor = config.coor
      if len(coor) > 0: self._x = clspt.Variable( np.array(coor[0]).astype(np.float32), read_only = True )
      if len(coor) > 1: self._y = clspt.Variable( np.array(coor[1]).astype(np.float32), read_only = True )
      if len(coor) > 2: self._z = clspt.Variable( np.array(coor[2]).astype(np.float32), read_only = True )

    self._temps = {}
    
    self.types = clspt.Variable( np.array(config.types).astype(np.uint8), read_only = True ) if config.types else None
    self.v = clspt.Variable( np.zeros(self.shape).astype(np.float32) )
    self.v_reset = config.fetch("v_reset", 0.0)
    self.v_thre = config.fetch("v_thre", 0.0)

    trefs = np.zeros(self.shape).astype(np.float32) + config.fetch("t_ref", 0.0)
    self.trefs = clspt.Variable( trefs, read_only=True )

    # for rk2 voltage evolving
    self.alpha0 = clspt.Variable( np.zeros(self.shape).astype(np.float32) )
    self.beta0 = clspt.Variable( np.zeros(self.shape).astype(np.float32) )
    self.alpha1 = clspt.Variable( np.zeros(self.shape).astype(np.float32) )
    self.beta1 = clspt.Variable( np.zeros(self.shape).astype(np.float32) )
    
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
    self.tspikes = np.zeros((self.nneurons,)).astype(np.float32)
    self.tspikes = clspt.Variable( self.tspikes )
    # Which neuron spiked
    self.ispikes = np.zeros((self.nneurons,)).astype(np.int32)
    self.ispikes = clspt.Variable( self.ispikes )

  def __getattr__(self, name):
    if name[0:4] == "temp":
      idx = name[4:]
      if not idx in self._temps: self._temps[idx] = clspt.Variable( shape=self.shape, dtype=np.float32 )
        return self._temps[idx]
    return object.__getattr__(self, name)

  def step(self, t, dt):
    pass
