# V1 neurons

from largescale.src.neuron import NeuronGroup, T_EXC, T_INH
import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import largescale.src.support.cl_common as clcom
import largescale.src.neuron.program as program
from largescale.src.support import CommonConfig
from largescale.src.noisy import NoisyConnection

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
    self.firing_rate_noisy_exc_nmda1 = config.fetch("firing_rate_noisy_exc_nmda1", 0.0)
    self.tau_rise_noisy_exc_nmda1 = config.fetch("tau_rise_noisy_exc_nmda1", 0.0)
    self.tau_damp_noisy_exc_nmda1 = config.fetch("tau_damp_noisy_exc_nmda1", 0.0)
    self.firing_rate_noisy_inh_gaba = config.fetch("firing_rate_noisy_inh_gaba", 0.0)
    self.tau_rise_noisy_inh_gaba = config.fetch("tau_rise_noisy_inh_gaba", 0.0)
    self.tau_damp_noisy_inh_gaba = config.fetch("tau_damp_noisy_inh_gaba", 0.0)
    self.firing_rate_noisy_exc_nmda2 = config.fetch("firing_rate_noisy_exc_nmda2", 0.0)
    self.tau_rise_noisy_exc_nmda2 = config.fetch("tau_rise_noisy_exc_nmda2", 0.0)
    self.tau_damp_noisy_exc_nmda2 = config.fetch("tau_damp_noisy_exc_nmda2", 0.0)
    self.firing_rate_noisy_inh_ampa = config.fetch("firing_rate_noisy_inh_ampa", 0.0)
    self.tau_rise_noisy_inh_ampa = config.fetch("tau_rise_noisy_inh_ampa", 0.0)
    self.tau_damp_noisy_inh_ampa = config.fetch("tau_damp_noisy_inh_ampa", 0.0)
    self.tau_rise_lgn_on = config.fetch("tau_rise_lgn_on", 0.0)
    self.tau_damp_lgn_on = config.fetch("tau_damp_lgn_on", 0.0)
    self.tau_rise_lgn_off = config.fetch("tau_rise_lgn_off", 0.0)
    self.tau_damp_lgn_off = config.fetch("tau_damp_lgn_off", 0.0)
    # self.tau_rise_gaba1 = config.fetch("tau_rise_gaba1", 0.0)
    # self.tau_damp_gaba1 = config.fetch("tau_damp_gaba1", 0.0)
    # self.tau_rise_gaba2 = config.fetch("tau_rise_gaba2", 0.0)
    # self.tau_damp_gaba2 = config.fetch("tau_damp_gaba2", 0.0)
    # self.tau_rise_ampa = config.fetch("tau_rise_ampa", 0.0)
    # self.tau_damp_ampa = config.fetch("tau_damp_ampa", 0.0)
    # self.tau_rise_nmda = config.fetch("tau_rise_nmda", 0.0)
    # self.tau_damp_nmda = config.fetch("tau_damp_nmda", 0.0)
    # self.g_leak = config.fetch("g_leak", 0.0)
    # self.kernels = config.fetch("kernels", None)
    # self.ikernel = clspt.Variable( config.fetch("ikernel", np.zeros(nshape).astype(np.double)), read_only = True )
    # self.g_lgn_on = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.s_lgn_on = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.g_lgn_off = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.s_lgn_off = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.g_ampa = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.s_ampa = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.g_nmda = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.s_nmda = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.g_gaba1 = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.s_gaba1 = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.g_gaba2 = clspt.Variable( np.zeros(nshape).astype(np.double) )
    # self.s_gaba2 = clspt.Variable( np.zeros(nshape).astype(np.double) )

    trefs[self.types == T_EXC] = self.t_ref_exc
    trefs[self.types == T_INH] = self.t_ref_inh
    self.trefs = clspt.Variable( trefs, read_only=True )

    self.lgn_kernel_on = config.fetch("lgn_kernel_on", None)
    self.lgn_kernel_off = config.fetch("lgn_kernel_off", None)
    self.noisy_exc_nmda1 = NoisyConnection(tau_rise = self.tau_rise_noisy_exc_nmda1, tau_damp = self.tau_damp_noisy_exc_nmda1, shape = self.shape, firing_rate = self.firing_rate_noisy_exc_nmda1)
    self.noisy_inh_gaba = NoisyConnection(tau_rise = self.tau_rise_noisy_inh_gaba, tau_damp = self.tau_damp_noisy_inh_gaba, shape = self.shape, firing_rate = self.firing_rate_noisy_inh_gaba)
    self.noisy_exc_nmda2 = NoisyConnection(tau_rise = self.tau_rise_noisy_exc_nmda2, tau_damp = self.tau_damp_noisy_exc_nmda2, shape = self.shape, firing_rate = self.firing_rate_noisy_exc_nmda2)
    self.noisy_inh_ampa = NoisyConnection(tau_rise = self.tau_rise_noisy_inh_ampa, tau_damp = self.tau_damp_noisy_inh_ampa, shape = self.shape, firing_rate = self.firing_rate_noisy_inh_ampa)

  def step(self, t, dt):
    NeuronGroup.step(self, t, dt)
    # program.chain2(self.g_lgn_on, self.s_lgn_on, self.tau_rise_lgn_on, self.tau_damp_lgn_on, dt)
    # program.chain2(self.g_lgn_off, self.s_lgn_off, self.tau_rise_lgn_off, self.tau_damp_lgn_off, dt)
    # program.chain2(self.g_gaba1, self.s_gaba1, self.tau_rise_gaba1, self.tau_damp_gaba1, dt)
    # program.chain2(self.g_gaba2, self.s_gaba2, self.tau_rise_gaba2, self.tau_damp_gaba2, dt)
    # program.chain2(self.g_ampa, self.s_ampa, self.tau_rise_ampa, self.tau_damp_ampa, dt)
    # program.chain2(self.g_nmda, self.s_nmda, self.tau_rise_nmda, self.tau_damp_nmda, dt)
    program.rk2voltage(self.v, self.tspikes, self.trefs, self.alpha0, self.beta0, self.alpha1, self.beta1, self.v_thre, self.v_reset, t, dt)
