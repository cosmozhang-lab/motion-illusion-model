# Neuron connection

import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support.common import CommonConfig
from largescale.src.neuron.neuron.program import chain2
from largescale.src.support.convolution import Conv2DKernel
from largescale.src.support.cl_support import ValuePoolSpec, ValuePool
import program




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
    self.cncts_host = np.zeros(np.sum([cnct.size for cnct in self.cncts])).astype(np.float32)
    self.cnct_shapes_dev = clspt.Variable( self.cnct_shapes_host, read_only = True )
    self.cncts_dev = clspt.Variable( self.cnct_shapes_host, read_only = True )



# Neuron connection
# Receives spikes of the input neuron group
# Gives out the conductance
class Connection:
  def __init__(self, config = None, **kwargs):
    if config is None: config = CommonConfig(kwargs)
    self.tau_rise = config.get("tau_rise", 0.0) # Rising time constants
    self.tau_damp = config.get("tau_damp", 0.0) # Damping time constants
    self.input = config.get("input", None) # Input neuron group
    self.shape = config.get("shape", None) # shape
    self.amp = config.get("amp", 1.0)
    self.amp_pool = config.get("amp_pool", ValuePool([self.amp], np.zeros(self.shape)))
    self.tau_rise_pool = config.get("tau_rise_pool", ValuePool([self.tau_rise], np.zeros(self.shape)))
    self.tau_damp_pool = config.get("tau_damp_pool", ValuePool([self.tau_damp], np.zeros(self.shape)))
    self.connectivity_pool = config.get("connectivity_pool", None) # connectivity pool
    if not isinstance(self.connectivity_pool, ConnectivityPool): self.connectivity_pool = ConnectivityPool(self.connectivity_pool)
    self.iconnectivities = config.get("iconnectivities", None) # indexes of connectivities to use
    if not isinstance(self.iconnectivities, clspt.Variable): self.iconnectivities = clspt.Variable(self.iconnectivities, read_only=True)
    self.kernel = config.get("kernel", None) # connection kernel
    if not isinstance(self.kernel, Conv2DKernel): self.kernel = Conv2DKernel(self.kernel)
    self.connection_map = config.get("connection_map", None) # if this is set, only neurons with same map index will be connected
    if self.connection_map and not isinstance(self.connection_map, clspt.Variable): self.connection_map = clspt.Variable(self.connection_map, read_only=True)
    self.g = clspt.Variable( np.zeros(self.shape).astype(np.float32) ) # conductance
    self.s = clspt.Variable( np.zeros(self.shape).astype(np.float32) ) # conductance relaxation
  def step(self, t, dt):
    last_t = t
    for i_nspikes in xrange(self.input.nspikes):
      tspk = self.input.tspikes.buf_host[i_nspikes]
      ispk = self.input.ispikes.buf_host[i_nspikes]
      if tspk > t and tspk <= t + dt:
        if tspk > last_t:
          chain2(self.g, self.s, self.tau_rise_pool, self.tau_damp_pool, tspk - last_t, update=True)
        input(self.s, ispk, self.amp_pool, self.kernel, self.connectivity_pool, self.iconnectivities, self.connection_map, self.tau_rise_pool, update=True)
        last_t = tspk
    if t + dt > last_t:
      chain2(self.g, self.s, self.tau_rise_pool, self.tau_damp_pool, t + dt - last_t, update=True)
      last_t = t + dt



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
def input(s, ispike, amp_pool, kernel, connectivity_pool, iconnectivities, connection_map, tau_rise_pool, queue=None, update=True):
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
    amp_pool.buf,
    amp_pool.spec.buf,
    iconnectivities.buf_dev,
    connectivity_pool.cnct_shapes_dev,
    connectivity_pool.cncts_dev,
    connection_map.buf_dev,
    s.buf_dev,
    s.buf_swp,
    tau_rise_pool.buf,
    tau_rise_pool.spec.buf
  )
  if update:
    s.update(queue)
