import pyopencl as cl
import largescale.src.support.cl_support as clspt
import numpy as np
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]

program_file = open( os.path.join(thisdir, "program.cl") )
program = cl.Program(clspt.context(), program_file.read()).build()
program_file.close()

# Calcuate the conductance decaying process
# The process can be expressed as ODEs:
#     tau_damp * dg/dt = - g + s
#     tau_rise * ds/dt = - s + sum( delta(t - t_spike) )
# @param g:        the conductances
# @param s:        the relaxation items (ds/dt receives the spike pulse directly)
# @param tau_rise: time constance of conductance rising
# @param tau_damp: time constance of conductance damping
# @param dt:       delta time
# [WARNING] This function updates the Variable buffer automatically!
kernelf_chain2 = program.chain2
kernelf_chain2.set_scalar_arg_dtypes([None, None, np.double, np.double, np.double])
def chain2(queue, g, s, tau_rise, tau_damp, dt):
  assert(g.shape == s.shape, "g and s must have the same shape")
  nneuron = g.size
  kernelf_chain2(queue, (nneuron,), None, g.buf_dev, s.buf_dev, tau_rise, tau_damp, dt)
