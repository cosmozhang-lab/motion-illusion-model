import pyopencl as cl
import largescale.src.support.cl_support as clspt
import numpy as np
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "program.cl") )

# Calcuate the conductance decaying process
# The process can be expressed as ODEs:
#     tau_damp * dg/dt = - g + s
#     tau_rise * ds/dt = - s + sum( delta(t - t_spike) )
# @param g:        [Variable]<double> the conductances
# @param s:        [Variable]<double> the relaxation items (ds/dt receives the spike pulse directly)
# @param tau_rise: [double] time constance of conductance rising
# @param tau_damp: [double] time constance of conductance damping
# @param dt:       [double] delta time
# @kwarg queue:    [CommandQueue]
# @kwarg update:   [Boolean] whether to update the variables immediately
# [WARNING] This function updates the Variable buffer automatically!
kernelf_chain2 = program.chain2
def chain2(g, s, tau_rise, tau_damp, dt, queue=None, update=True):
  assert(g.shape == s.shape, "g and s must have the same shape")
  nneuron = g.size
  kernelf_chain2(queue, (nneuron,), None, g.buf_dev, g.buf_swp, s.buf_dev, s.buf_swp, tau_rise, tau_damp, dt, np.exp(dt/tau_rise), np.exp(dt/tau_damp))
  if update:
    g.update(queue)
    s.update(queue)
