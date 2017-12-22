# V1 neurons

from largescale.src.neuron import NeuronGroup, T_EXC, T_INH
import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import largescale.src.support.cl_common as clcom
from program import rk2params
import largescale.src.neuron.program as program
from largescale.src.support.common import CommonConfig
from largescale.src.neuron.connection import NoisyConnection

# V1 neuron group that directly receive stimulus (skipped LGN)
class V1DirectNeuronGroup (NeuronGroup):
  # config:
  #     ** inherit from NeuronGroup.config
  #     type: neuron types as np.array(uint8)
  #     stimulus: an stimulus sorce (instance of StimulusBase)
  #     v_lgn: the resting voltage for LGN inputs
  #     v_exc: the resting voltage for excitatory
  #     v_inh: the resting voltage for inhibitory
  #     tau_rise_lgn_on: the time constant of conductance rising for LGN ON pathway
  #     tau_damp_lgn_on: the time constant of conductance damping for LGN ON pathway
  #     tau_rise_lgn_off: the time constant of conductance rising for LGN OFF pathway
  #     tau_damp_lgn_off: the time constant of conductance damping for LGN OFF pathway
  #     tau_rise_gaba1: the time constant of conductance rising for gaba1
  #     tau_damp_gaba1: the time constant of conductance damping for gaba1
  #     tau_rise_gaba2: the time constant of conductance rising for gaba2
  #     tau_damp_gaba2: the time constant of conductance damping for gaba2
  #     tau_rise_ampa: the time constant of conductance rising for ampa
  #     tau_damp_ampa: the time constant of conductance damping for ampa
  #     tau_rise_nmda: the time constant of conductance rising for nmda
  #     tau_damp_nmda: the time constant of conductance damping for nmda
  #     g_leak: the leaking conductance as a constance
  def __init__(self, nshape, config=None):
    NeuronGroup.__init__(self, nshape, config = CommonConfig())
    self.stimulus = config.fetch("stimulus")
    self.v_exc = config.fetch("v_exc", 0.0)
    self.v_inh = config.fetch("v_inh", 0.0)
    self.t_ref_exc = config.fetch("t_ref_exc", 0.0)
    self.t_ref_inh = config.fetch("t_ref_inh", 0.0)
    self.fgaba = config.fetch("fgaba", 0.0) # gaba and ampa for inhibitory
    self.fnmda = config.fetch("fnmda", 0.0) # nmda1 and nmda2 for excitatory
    self.fgaba_noise = config.fetch("fgaba_noise", 0.0) # gaba and ampa for inhibitory of noise
    self.fnmda_noise = config.fetch("fnmda_noise", 0.0) # nmda1 and nmda2 for excitatory of noise
    self.g_leak = config.fetch("g_leak", 0.0)
    self.firing_rate_noisy_exc_nmda = config.fetch("firing_rate_noisy_exc_nmda", 0.0)
    self.tau_rise_noisy_exc_nmda = config.fetch("tau_rise_noisy_exc_nmda", 0.0)
    self.tau_damp_noisy_exc_nmda = config.fetch("tau_damp_noisy_exc_nmda", 0.0)
    self.firing_rate_noisy_inh_gaba1 = config.fetch("firing_rate_noisy_inh_gaba1", 0.0)
    self.tau_rise_noisy_inh_gaba1 = config.fetch("tau_rise_noisy_inh_gaba1", 0.0)
    self.tau_damp_noisy_inh_gaba1 = config.fetch("tau_damp_noisy_inh_gaba1", 0.0)
    self.firing_rate_noisy_exc_ampa = config.fetch("firing_rate_noisy_exc_ampa", 0.0)
    self.tau_rise_noisy_exc_ampa = config.fetch("tau_rise_noisy_exc_ampa", 0.0)
    self.tau_damp_noisy_exc_ampa = config.fetch("tau_damp_noisy_exc_ampa", 0.0)
    self.firing_rate_noisy_inh_gaba2 = config.fetch("firing_rate_noisy_inh_gaba2", 0.0)
    self.tau_rise_noisy_inh_gaba2 = config.fetch("tau_rise_noisy_inh_gaba2", 0.0)
    self.tau_damp_noisy_inh_gaba2 = config.fetch("tau_damp_noisy_inh_gaba2", 0.0)
    self.tau_rise_lgn_on = config.fetch("tau_rise_lgn_on", 0.0)
    self.tau_damp_lgn_on = config.fetch("tau_damp_lgn_on", 0.0)
    self.tau_rise_lgn_off = config.fetch("tau_rise_lgn_off", 0.0)
    self.tau_damp_lgn_off = config.fetch("tau_damp_lgn_off", 0.0)
    # short-range recurrent connections
    self.recurrent_connectivities = config.fetch("recurrent_connectivities", None)
    self.recurrent_iconnectivities = config.fetch("recurrent_iconnectivities", None)
    self.recurrent_kernel_exc_exc = config.fetch("recurrent_kernel_exc_exc", None)
    self.recurrent_kernel_inh_exc = config.fetch("recurrent_kernel_inh_exc", None)
    self.recurrent_kernel_exc_inh = config.fetch("recurrent_kernel_exc_inh", None)
    self.recurrent_kernel_inh_inh = config.fetch("recurrent_kernel_inh_inh", None)
    # long-range recurrent connections
    self.recurrent_connectivities_lr = config.fetch("recurrent_connectivities_lr", None)
    self.recurrent_iconnectivities_lr = config.fetch("recurrent_iconnectivities_lr", None)
    self.recurrent_kernel_exc_exc_lr = config.fetch("recurrent_kernel_exc_exc_lr", None)
    self.recurrent_kernel_inh_exc_lr = config.fetch("recurrent_kernel_inh_exc_lr", None)

    trefs[self.types == T_EXC] = self.t_ref_exc
    trefs[self.types == T_INH] = self.t_ref_inh
    self.trefs = clspt.Variable( trefs, read_only=True )

    self.lgn_kernels_on = config.fetch("lgn_kernels_on", None)
    self.lgn_ikernels_on = config.fetch("lgn_ikernels_on", None)
    self.lgn_kernels_off = config.fetch("lgn_kernels_off", None)
    self.lgn_ikernels_off = config.fetch("lgn_ikernels_off", None)
    self.lgn_on = DirectConnection(tau_rise = self.tau_rise_lgn_on, tau_damp = self.tau_damp_lgn_on, shape = self.shape, stimulus = self.stimulus, kernels = self.lgn_kernels_on, ikernels = self.lgn_ikernels_on)
    self.lgn_off = DirectConnection(tau_rise = self.tau_rise_lgn_off, tau_damp = self.tau_damp_lgn_off, shape = self.shape, stimulus = self.stimulus, kernels = self.lgn_kernels_off, ikernels = self.lgn_ikernels_off)
    self.noisy_exc_nmda = NoisyConnection(tau_rise = self.tau_rise_noisy_exc_nmda, tau_damp = self.tau_damp_noisy_exc_nmda, shape = self.shape, firing_rate = self.firing_rate_noisy_exc_nmda)
    self.noisy_inh_gaba1 = NoisyConnection(tau_rise = self.tau_rise_noisy_inh_gaba1, tau_damp = self.tau_damp_noisy_inh_gaba1, shape = self.shape, firing_rate = self.firing_rate_noisy_inh_gaba1)
    self.noisy_exc_ampa = NoisyConnection(tau_rise = self.tau_rise_noisy_exc_ampa, tau_damp = self.tau_damp_noisy_exc_ampa, shape = self.shape, firing_rate = self.firing_rate_noisy_exc_ampa)
    self.noisy_inh_gaba2 = NoisyConnection(tau_rise = self.tau_rise_noisy_inh_gaba2, tau_damp = self.tau_damp_noisy_inh_gaba2, shape = self.shape, firing_rate = self.firing_rate_noisy_inh_gaba2)

  def step(self, t, dt):
    NeuronGroup.step(self, t, dt)
    program.rk2params(self.lgn_on.g, self.lgn_off.g, self.noisy_inh_gaba1, self.noisy_inh_gaba2, self.noisy_exc_nmda, self.noisy_exc_ampa, self.fgaba_noise, slef.fnmda_noise, self.alpha0, self.beta0)
    self.lgn_on.step(t, dt)
    self.lgn_off.step(t, dt)
    self.noisy_exc_nmda.step(t, dt)
    self.noisy_inh_gaba1.step(t, dt)
    self.noisy_exc_ampa.step(t, dt)
    self.noisy_inh_gaba2.step(t, dt)
    program.rk2params(self.lgn_on.g, self.lgn_off.g, self.noisy_inh_gaba1, self.noisy_inh_gaba2, self.noisy_exc_nmda, self.noisy_exc_ampa, self.fgaba_noise, slef.fnmda_noise, self.alpha1, self.beta1)
    program.rk2voltage(self.v, self.tspikes, self.trefs, self.alpha0, self.beta0, self.alpha1, self.beta1, self.v_thre, self.v_reset, t, dt)


