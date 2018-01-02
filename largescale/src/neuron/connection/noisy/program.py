import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support.common import CommonConfig
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "program.cl") )

"""
Calcuate the conductance decaying process
for noisy input group. Each neuron may
spike several times in `dt`. We simulate
this process by taking following process:
    `chain2` on `t` to `t_spike_1`
    `s` = `s` + `amp` / `tau_r`
    `chain2` on `t_spike_1` to `t_spike_2`
    `s` = `s` + `amp` / `tau_r`
    ...
    `chain2` on `t_spike_last` to `t + dt`
The `chain2` process can be expressed as ODEs:
    tau_damp * dg/dt = - g + s
    tau_rise * ds/dt = - s
The spikes stream obeys a poisson progress: 
    P[N(t+tau)-N(t)=k] = 
        exp(-lambda*tau) * (lambda*tau)^k / k!
So the spike interval `tau` obeys distribution:
    pdf(tau) = P[N(t+tau)-N(t)=k]
             = (lambda*tau) * exp(-lambda*tau)
@param g:           the conductances
@param s:           the relaxation items (ds/dt receives the spike pulse directly)
@param tspikes:     spiking times (the buffer is used for both read and write)
@param firing_rate: the noisy firing rate
@param tau_rise:    time constance of conductance rising
@param tau_damp:    time constance of conductance damping
@param t:           start time
@param dt:          delta time
@param randseeds:   random seed
"""
kernel_chain2noisy = program.chain2noisy.kernel
def chain2noisy(g, s, tspikes, amp_pool, firing_rate_pool, tau_rise_pool, tau_damp_pool, t, dt, randseeds = None, queue = None, update = False):
  assert g.shape == s.shape, "g and s must have the same shape"
  assert g.shape == tspikes.shape, "g and tspikes must have the same shape"
  if randseeds is None:
    randseeds = clspt.Variable( np.floor(np.random.random_sample(g.shape) * (clspt.RAND_MAX+1)).astype(np.float32), read_only = True )
  assert g.shape == randseeds.shape, "g and tspikes must have the same shape"
  if queue is None: queue = clspt.queue()
  nneurons = np.prod(g.shape)
  kernel_chain2noisy(queue, (nneurons,), None,
    g.buf_dev,
    g.swp_dev,
    s.buf_dev,
    s.swp_dev,
    tspikes.buf_dev,
    amp_pool.buf,
    amp_pool.spec.buf,
    firing_rate_pool.buf,
    firing_rate_pool.spec.buf,
    tau_rise_pool.buf,
    tau_rise_pool.spec.buf,
    tau_damp_pool.buf,
    tau_damp_pool.spec.buf,
    t,
    dt,
    randseeds.buf_dev)
  if update:
    g.update(queue)
    s.update(queue)
    tspikes.update(queue)
    randseeds.update(queue)
