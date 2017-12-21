# Connection from received stimulus to neuron

import numpy as np
import largescale.src.support.cl_support as clspt
import largescale.src.support.cl_common as clcom
from largescale.src.support import CommonConfig
from largescale.src.neuron import chain2
from largescale.src.convolution import Conv2DKernelPool, conv2d
from connection import Connection

class DirectConnection (Connection):
  def __init__(self, config = None, **kwargs):
    if config is None: config = CommonConfig(kwargs)
    Connection.__init__(self, config)
    self.stimulus = config.get("stimulus", None)
    self.kernel = config.get("kernel", None)
    self.kernels = Conv2DKernelPool([self.kernel])
    self.ikernels = clspt.Variable( np.zeros(self.stimulus.size).astype(np.int32), read_only=True )
    self.stibuf = clspt.Variable( np.zeros(self.stimulus.size).astype(np.double), auto_update=True )
    self.convbuf = clspt.Variable( np.zeros(self.stimulus.size).astype(np.double), auto_update=True )
  def step(self, t, dt):
    self.stimulus.get(t, var = self.stibuf)
    conv2d(self.stibuf, self.convbuf, self.kernels, self.ikernels, update=True)
    clcom.add(self.convbuf, self.s, self.s, update=True)
    chain2(self.g, self.s, self.tau_rise, self.tau_damp, dt, update=True)


