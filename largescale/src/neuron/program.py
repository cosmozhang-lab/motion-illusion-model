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
kernelf_chain2 = program.chain2.kernel
def chain2(g, s, tau_rise, tau_damp, dt, queue=None, update=True):
  assert(g.shape == s.shape, "g and s must have the same shape")
  nneuron = g.size
  kernelf_chain2(queue, (nneuron,), None, g.buf_dev, g.buf_swp, s.buf_dev, s.buf_swp, tau_rise, tau_damp, dt, np.exp(dt/tau_rise), np.exp(dt/tau_damp))
  if update:
    g.update(queue)
    s.update(queue)






"""
Use RK2 algorithm to calculate the voltage
evolution and the spikes.
Voltage evolves according to ODE:
  dv/dt = sum<i>{ - g_i * (v - v_ref_i) }
  Where `i` represents either `leak`, `in-
  hibitory` or `excitatory`. `v_ref_i` re-
  presents the refactory votage.
  and:
    g_inh = g_inh_recurrent + g_inh_noisy
    g_exc = g_exc_recurrent + g_exc_noisy
            + g_input
As we use RK2 algorithm, `v(t+dt)` can be
calculated as:
  v(t+dt) = v(t) + k * dt
  where:
  k = ( d{v(t)}/dt + {d{v(t)+k1*dt}/dt}_t ) / 2
  where:
  k1 = {dv/dt}_t
     = sum<i>{ - g_i(t) * (v(t) - v_ref_i) }
  k2 = {dv/dt}_t
     = sum<i>{ - g_i(t+dt) * (v(t) + k1*dt - v_ref_i) }
Note that if neuron spiked at `ts`, then 
voltage will be reset to a refactory vo-
ltage `v_reset`, and hold this level for
at least a refactory time `t_ref`. And
also note that neuron spikes immediately
when `v` reaches threshold `v_thre`. So
we must:
1) Evolve `v` from time point `ts+t_ref`
   if it is between `t` and `t+dt`, ins-
   tead of evolving from `t`.
2) reset `v` to `v_reset` at `ts` if `ts`
   is between `t` and `t+dt`. In which:
     ts = t + (v_thre - v) / k
   And then hold `v_reset` until `ts +
   t_ref` or to the end of the time bin
   `t + dt`.
As we will only have `g_i` at the start
and the end of the time bin, according
to the conductance evolving stage, we
cannot make it so precise that the evo-
lving start time is calculated conside-
ring all the above conditions. So for
the `g_i`s in the equations we can only
take them as:
  g_i(t) = gi0
  g_i(t+dt) = gi1
where `gi0` represents `g_i` at the
start of the time bin, and `gi1` that
at the end of the time bin.
Besides, to reduce the number of params
of this RK2 function, we transform the
equations as:
  k1 = - sum<i>{ gi0 } * v(t) + sum<i>{ gi0 * v_ref_i }
  k1 = - sum<i>{ gi1 } * (v(t)+k1*dt) + sum<i>{ gi1 * v_ref_i }
so we can represent all the conductance
params (i.e. `gi`s) into four params:
  alpha0 = - sum<i>{ gi0 }
  beta0 = sum<i>{ gi0 * v_ref_i }
  alpha1 = - sum<i>{ gi1 }
  beta1 = sum<i>{ gi1 * v_ref_i }
@param v:                   voltage of each neuron
@param tspikes:             last spike times
@param t_refs:              refactory time for each neuron
@param v_thre:              voltage spike threshold
@param v_reset:             voltage in refactory period
@param alpha0:              alpha params at `t`
@param beta0:               beta params at `t`
@param alpha1:              alpha params at `t` + `dt`
@param beta1:               beta params at `t` + `dt`
@param t:                   time bin start
@param dt:                  delta time
@kwarg queue:               [CommandQueue]
@kwarg update:              [Boolean] whether to update the variables immediately
"""
kernel_rk2voltage = program.rk2voltage
def rk2voltage(v, tspikes, trefs, alpha0, beta0, alpha1, beta1, v_thre, v_reset, t, dt, queue=None, update=True):
  queue = queue or clspt.queue()
  kernel_rk2voltage(queue, (v.size,), None, v.buf_dev, v.buf_swp, tspikes.buf_dev, trefs.buf_dev, alpha0.buf_dev, beta0.buf_dev, alpha1.buf_dev, beta1.buf_dev, v_thre, v_reset, t, dt)
  if update:
    v.update()
    tspikes.update()
