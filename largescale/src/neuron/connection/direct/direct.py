# Connection from received stimulus to neuron

import numpy as np
import largescale.src.support.cl_support as clspt
import largescale.src.support.cl_common as clcom
from largescale.src.support.common import CommonConfig
from largescale.src.neuron.neuron.program import chain2
from largescale.src.support.convolution import Conv2DKernelPool, conv2d
from largescale.src.neuron.connection import Connection

class DirectConnection (Connection):
  def __init__(self, config = None, **kwargs):
    if config is None: config = CommonConfig(kwargs)
    Connection.__init__(self, config)
    self.stimulus = config.get("stimulus", None)
    self.kernels = config.get("kernels", None)
    if not isinstance(self.kernels, Conv2DKernelPool): self.kernels = Conv2DKernelPool( self.kernels )
    self.ikernels = config.get("kernels", None)
    if not isinstance(self.ikernels, clspt.Variable): self.ikernels = clspt.Variable( self.ikernels.astype(np.int32), read_only=True )
    self.stibuf = clspt.Variable( np.zeros(self.stimulus.size).astype(np.float32), auto_update=True )
    self.convbuf = clspt.Variable( np.zeros(self.stimulus.size).astype(np.float32), auto_update=True )
  def step(self, t, dt):
    self.stimulus.get(t, var = self.stibuf)
    conv2d(self.stibuf, self.convbuf, self.kernels, self.ikernels, update=True)
    clcom.add(self.convbuf, self.s, self.s, update=True)
    chain2(self.g, self.s, self.tau_rise_pool, self.tau_damp_pool, dt, update=True)


