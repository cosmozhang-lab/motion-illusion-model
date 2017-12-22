# Neuron connection

import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support import CommonConfig
from largescale.src.neuron import chain2
from largescale.src.convolution import Conv2DKernel
import program
import os




# Besides the global kernel, we use connectivity pool to build
# random connections. A connectivity pool contains several
# connectivity matrixes. Each is a connectivity pool centered 
# on the output neuron.
class ConnectivityPool:
  # @param connectivities
  def __init__(self, connectivities):
    self.num = len(connectivities)
    for cnct in connectivities:
      if not isinstance(cnct, np.ndarray) or not cnct.ndim == 2:
        raise ValueError("Connectivity shuold be a 2D matrix!")
    self.cncts = connectivities
    self.cnct_shapes_host = np.zeros(2*self.num).astype(np.int32)
    self.cncts_host = np.zeros(np.sum([cnct.size for cnct in self.cncts])).astype(np.double)
    self.cnct_shapes_dev = clspt.Variable( self.cnct_shapes_host, read_only = True )
    self.cncts_dev = clspt.Variable( self.cnct_shapes_host, read_only = True )



# Neuron connection
# Receives spikes of the input neuron group
# Gives out the conductance
class Connection:
  def __init__(self, config = None, **kwargs):
    if config is None: config = CommonConfig(kwargs)
    self.tau_rise = config.get("tau_rise", 0.0) # Rising time constant
    self.tau_damp = config.get("tau_damp", 0.0) # Damping time constant
    self.input = config.get("input", None) # Input neuron group
    self.shape = config.get("shape", None) # shape
    self.connectivity_pool = config.get("connectivity_pool", None) # connectivity pool
    if not isinstance(self.connectivity_pool, ConnectivityPool): self.connectivity_pool = ConnectivityPool(self.connectivity_pool)
    self.iconnectivities = config.get("iconnectivities", None) # indexes of connectivities to use
    if not isinstance(self.iconnectivities, clspt.Variable): self.iconnectivities = clspt.Variable(self.iconnectivities, read_only=True)
    self.kernel = config.get("kernel", None) # connection kernel
    if not isinstance(self.kernel, Conv2DKernel): self.kernel = Conv2DKernel(self.kernel)
    self.g = clspt.Variable( np.zeros(self.shape).astype(np.double) ) # conductance
    self.s = clspt.Variable( np.zeros(self.shape).astype(np.double) ) # conductance relaxation
  def step(self, t, dt):
    last_t = t
    for i_nspikes in xrange(self.input.nspikes):
      tspk = self.input.tspikes.buf_host[i_nspikes]
      ispk = self.input.ispikes.buf_host[i_nspikes]
      if tspk > t and tspk <= t + dt:
        if tspk > last_t:
          chain2(self.g, self.s, self.tau_rise, self.tau_damp, tspk - last_t, update=True)
        input(self.s, ispk, self.kernel, self.connectivity_pool, self.iconnectivities, self.tau_rise, update=True)
        last_t = tspk
    if t + dt > last_t:
      chain2(self.g, self.s, self.tau_rise, self.tau_damp, t + dt - last_t, update=True)
      last_t = t + dt




"""
Calcuate the conductance decaying process with spike input.
This process do involve spike inputs,
which is different from `chain2` in `neuron/program.cl`.
The process can be expressed as ODEs:
    tau_damp * dg/dt = - g + s
    tau_rise * ds/dt = - s + sum( delta(t - t_spike) )
@Remark: I don't think this is necessary. As we can use
chain2 without input, just add `amp` / tau_r to `s` on each
spike. This does make sense. The process is:
    `chain2` on `t` to `t_spike_1`
    `s` = `s` + `amp` / `tau_r`
    `chain2` on `t_spike_1` to `t_spike_2`
    `s` = `s` + `amp` / `tau_r`
    ...
    `chain2` on `t_spike_last` to `t + dt`
@param g:                 [Variable]<double> the conductances
@param s:                 [Variable]<double> the relaxation items (ds/dt receives the spike pulse directly)
@param ispike:            [int] which neuron spiked
@param kernel:            [Conv2DKernel] the input mapping kernel
@param connectivity_pool: [ConnectivityPool] the input connectivity mapping pool
@param iconnectivities:   [Variable]<int> indexes of the connectivity mapping to use
@param tau_rise:          [double] time constance of conductance rising
@param tau_damp:          [double] time constance of conductance damping
@param dt:                [double] delta time
@kwarg queue:             [CommandQueue]
@kwarg update:            [Boolean] whether to update the variables immediately
[WARNING] This function updates the Variable buffer automatically!
"""
def chain2_with_input(g, s, ispike, kernel, connectivity_pool, iconnectivities, tau_rise, tau_damp, dt, queue=None, update=True):
  if queue is None: queue = clspt.queue()
  assert(g.shape == s.shape, "g and s must have the same shape")
  assert(g.shape == iconnectivities.shape, "g and iconnectivities must have the same shape")
  nneurons = g.size
  program.kernel_chain2_with_input(queue, (nneurons,), None,
    g.shape[0],
    g.shape[1],
    ispike,
    kernel.shape[0],
    kernel.shape[1],
    kernel.kernel_dev,
    iconnectivities.buf_dev,
    connectivity_pool.cnct_shapes_dev,
    connectivity_pool.cncts_dev,
    g.buf_dev,
    g.buf_swp,
    s.buf_dev,
    s.buf_swp,
    tau_rise,
    tau_damp,
    dt,
    np.exp(-dt / tau_rise),
    np.exp(-dt / tau_damp)
  )
  if update:
    g.update(queue)
    s.update(queue)




"""
Add spike input to relaxation item `s`. This function 
cooperate with `chain2`. In each `dt`, if there are
spikes in `dt, we should do `chain2` on each interval
between each two contiguous spikes. And between each
two contiguous `chain2` processes, we add the spike
input to relaxation item with this function(`input`).
The process is described as:
    `chain2` on `t` to `t_spike_1`
    `input` for `spike_1` ( `s` = `s` + `amp` / `tau_r` )
    `chain2` on `t_spike_1` to `t_spike_2`
    `input` for `spike_2` ( `s` = `s` + `amp` / `tau_r` )
    ...
    `chain2` on `t_spike_last` to `t + dt`

@param s:                 the relaxation items (ds/dt receives the spike pulse directly)
@param ispike:            [int] which neuron spiked
@param kernel:            [Conv2DKernel] the input mapping kernel
@param connectivity_pool: [ConnectivityPool] the input connectivity mapping pool
@param iconnectivities:   [Variable]<int> indexes of the connectivity mapping to use
@param tau_rise:          time constance of conductance rising
@kwarg queue:             [CommandQueue]
@kwarg update:            [Boolean] whether to update the variables immediately
[WARNING] This function updates the Variable buffer automatically!
"""
def input(s, ispike, kernel, connectivity_pool, iconnectivities, tau_rise, queue=None, update=True):
  if queue is None: queue = clspt.queue()
  assert(s.shape == iconnectivities.shape, "s and iconnectivities must have the same shape")
  nneurons = s.size
  program.kernel_input(queue, (nneurons,), None,
    s.shape[0],
    s.shape[1],
    ispike,
    kernel.shape[0],
    kernel.shape[1],
    kernel.kernel_dev,
    iconnectivities.buf_dev,
    connectivity_pool.cnct_shapes_dev,
    connectivity_pool.cncts_dev,
    s.buf_dev,
    s.buf_swp,
    tau_rise
  )
  if update:
    s.update(queue)
