# Noisy input (a special input)
# Noisy input simulates a group of the background noisy inputs.
# For each neuron, we represent this as an integrated input from
# many other uninterested neuron groups. So each neuron has only
# one noisy input, which behaves like a noisy population. This 
# means that each neuron may receive more than one spikes in a
# single time bin.
# This is different to our common hypothsis in two aspects:
# 1) Each neuron receives only one noisy input, instead of rece-
#    iving inputs from many neurons in an input neuron group.
# 2) Each noisy input many spike more than once in a single time
#    bin. While in common, each input neuron spikes at most once
#    in a single time bin.
# In consideration of these features, we here implement an noisy
# input group as a kind of connection group. But this connection 
# receives no input, and gives out a noisy conductance on itself.

import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support import CommonConfig
from largescale.src.neuron import chain2
from largescale.src.convolution import Conv2DKernel
from largescale.src.connection import Connection
import program
import os

class NoisyConnection (Connection):
  def __init__(self, config = None, **kwargs):
    if config is None: config = CommonConfig(kwargs)
    Connection.__init__(self, config)
    self.firing_rate = config.get("firing_rate", 0.0)
    self.tspikes = clspt.Variable( np.zeros(self.shape).astype(np.double), auto_update = True )
    self.randseeds = clspt.Variable( np.zeros(self.shape).astype(np.double), auto_update = True )
  def step(self, t, dt):
    self.randseeds.fill( np.floor(np.random.random_sample(g.shape) * (clspt.RAND_MAX+1)).astype(np.double) )
    program.chain2noisy(self.g, self.s, self.tspikes, self.firing_rate, self.tau_rise, self.tau_damp, t, dt, self.randseeds)
