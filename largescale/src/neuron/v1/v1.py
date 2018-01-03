# V1 neurons

from largescale.src.neuron import NeuronGroup, T_EXC, T_INH
import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import largescale.src.support.cl_common as clcom
from program import rk2params
from largescale.src.neuron.neuron.program import rk2voltage
from largescale.src.support.common import CommonConfig
from largescale.src.neuron.connection import NoisyConnection
from largescale.src.support.cl_support import ValuePoolSpec, ValuePool

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
    self.fgaba = config.fetch("fgaba", 0.0) # gaba1 / (gaba1 + gaba2) for inhibitory
    self.fnmda = config.fetch("fnmda", 0.0) # nmda / (nmda + ampa) for excitatory
    self.fgaba_noise = config.fetch("fgaba_noise", 0.0) # gaba1 / (gaba1 + gaba2) for inhibitory of noise
    self.fnmda_noise = config.fetch("fnmda_noise", 0.0) # nmda / (nmda + ampa) for excitatory of noise
    self.g_leak = config.fetch("g_leak", 0.0)
    self.amp_noisy_exc_nmda = config.fetch("amp_noisy_exc_nmda", 0.0)
    self.firing_rate_noisy_exc_nmda = config.fetch("firing_rate_noisy_exc_nmda", 0.0)
    self.tau_rise_noisy_exc_nmda = config.fetch("tau_rise_noisy_exc_nmda", 0.0)
    self.tau_damp_noisy_exc_nmda = config.fetch("tau_damp_noisy_exc_nmda", 0.0)
    self.amp_noisy_exc_ampa = config.fetch("amp_noisy_exc_ampa", 0.0)
    self.firing_rate_noisy_exc_ampa = config.fetch("firing_rate_noisy_exc_ampa", 0.0)
    self.tau_rise_noisy_exc_ampa = config.fetch("tau_rise_noisy_exc_ampa", 0.0)
    self.tau_damp_noisy_exc_ampa = config.fetch("tau_damp_noisy_exc_ampa", 0.0)
    self.amp_noisy_exc_gaba1 = config.fetch("amp_noisy_exc_gaba1", 0.0)
    self.firing_rate_noisy_exc_gaba1 = config.fetch("firing_rate_noisy_exc_gaba1", 0.0)
    self.tau_rise_noisy_exc_gaba1 = config.fetch("tau_rise_noisy_exc_gaba1", 0.0)
    self.tau_damp_noisy_exc_gaba1 = config.fetch("tau_damp_noisy_exc_gaba1", 0.0)
    self.amp_noisy_exc_gaba2 = config.fetch("amp_noisy_exc_gaba2", 0.0)
    self.firing_rate_noisy_exc_gaba2 = config.fetch("firing_rate_noisy_exc_gaba2", 0.0)
    self.tau_rise_noisy_exc_gaba2 = config.fetch("tau_rise_noisy_exc_gaba2", 0.0)
    self.tau_damp_noisy_exc_gaba2 = config.fetch("tau_damp_noisy_exc_gaba2", 0.0)
    self.amp_noisy_inh_nmda = config.fetch("amp_noisy_inh_nmda", 0.0)
    self.firing_rate_noisy_inh_nmda = config.fetch("firing_rate_noisy_inh_nmda", 0.0)
    self.tau_rise_noisy_inh_nmda = config.fetch("tau_rise_noisy_inh_nmda", 0.0)
    self.tau_damp_noisy_inh_nmda = config.fetch("tau_damp_noisy_inh_nmda", 0.0)
    self.amp_noisy_inh_ampa = config.fetch("amp_noisy_inh_ampa", 0.0)
    self.firing_rate_noisy_inh_ampa = config.fetch("firing_rate_noisy_inh_ampa", 0.0)
    self.tau_rise_noisy_inh_ampa = config.fetch("tau_rise_noisy_inh_ampa", 0.0)
    self.tau_damp_noisy_inh_ampa = config.fetch("tau_damp_noisy_inh_ampa", 0.0)
    self.amp_noisy_inh_gaba1 = config.fetch("amp_noisy_inh_gaba1", 0.0)
    self.firing_rate_noisy_inh_gaba1 = config.fetch("firing_rate_noisy_inh_gaba1", 0.0)
    self.tau_rise_noisy_inh_gaba1 = config.fetch("tau_rise_noisy_inh_gaba1", 0.0)
    self.tau_damp_noisy_inh_gaba1 = config.fetch("tau_damp_noisy_inh_gaba1", 0.0)
    self.amp_noisy_inh_gaba2 = config.fetch("amp_noisy_inh_gaba2", 0.0)
    self.firing_rate_noisy_inh_gaba2 = config.fetch("firing_rate_noisy_inh_gaba2", 0.0)
    self.tau_rise_noisy_inh_gaba2 = config.fetch("tau_rise_noisy_inh_gaba2", 0.0)
    self.tau_damp_noisy_inh_gaba2 = config.fetch("tau_damp_noisy_inh_gaba2", 0.0)
    self.lgn_kernels_on = config.fetch("lgn_kernels_on", None)
    self.amp_lgn_on_pos = config.fetch("amp_lgn_on_pos", 0.0)
    self.tau_rise_lgn_on_pos = config.fetch("tau_rise_lgn_on_pos", 0.0)
    self.tau_damp_lgn_on_pos = config.fetch("tau_damp_lgn_on_pos", 0.0)
    self.lgn_ikernels_on = config.fetch("lgn_ikernels_on", None)
    self.amp_lgn_on_neg = config.fetch("amp_lgn_on_neg", 0.0)
    self.tau_rise_lgn_on_neg = config.fetch("tau_rise_lgn_on_neg", 0.0)
    self.tau_damp_lgn_on_neg = config.fetch("tau_damp_lgn_on_neg", 0.0)
    self.lgn_kernels_off = config.fetch("lgn_kernels_off", None)
    self.amp_lgn_off_pos = config.fetch("amp_lgn_off_pos", 0.0)
    self.tau_rise_lgn_off_pos = config.fetch("tau_rise_lgn_off_pos", 0.0)
    self.tau_damp_lgn_off_pos = config.fetch("tau_damp_lgn_off_pos", 0.0)
    self.lgn_ikernels_off = config.fetch("lgn_ikernels_off", None)
    self.amp_lgn_off_neg = config.fetch("amp_lgn_off_neg", 0.0)
    self.tau_rise_lgn_off_neg = config.fetch("tau_rise_lgn_off_neg", 0.0)
    self.tau_damp_lgn_off_neg = config.fetch("tau_damp_lgn_off_neg", 0.0)
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

    def two_type_var(exc_val, inh_val, dtype=np.float32, read_only=True):
      arr = np.zeros(self.shape).astype(dtype)
      arr[self.types == T_EXC] = exc_val
      arr[self.types == T_INH] = inh_val
      return clspt.Variable(arr, read_only=read_only)

    # Specs for the pools that is determined by Excitatory(0)/Inhibitory(1)
    self.type_spec = np.zeros(self.shape).astype(np.int32)
    self.type_spec[self.types == T_EXC] = 0
    self.type_spec[self.types == T_INH] = 1
    self.type_spec = ValuePoolSpec( self.type_spec )
    def two_type_pool(exc_val, inh_val, dtype=np.float32):
      return ValuePool(np.array([exc_val, inh_val]).astype(dtype), self.type_spec)

    self.trefs = two_type_pool(self.t_ref_exc, self.t_ref_inh)

    self.amp_noisy_nmda = two_type_pool(self.amp_noisy_exc_nmda, self.amp_noisy_inh_nmda)
    self.firing_rate_noisy_nmda = two_type_pool(self.firing_rate_noisy_exc_nmda, self.firing_rate_noisy_inh_nmda)
    self.tau_rise_noisy_nmda = two_type_pool(self.tau_rise_noisy_exc_nmda, self.tau_rise_noisy_inh_nmda)
    self.tau_damp_noisy_nmda = two_type_pool(self.tau_damp_noisy_exc_nmda, self.tau_damp_noisy_inh_nmda)
    self.amp_noisy_ampa = two_type_pool(self.amp_noisy_exc_ampa, self.amp_noisy_inh_ampa)
    self.firing_rate_noisy_ampa = two_type_pool(self.firing_rate_noisy_exc_ampa, self.firing_rate_noisy_inh_ampa)
    self.tau_rise_noisy_ampa = two_type_pool(self.tau_rise_noisy_exc_ampa, self.tau_rise_noisy_inh_ampa)
    self.tau_damp_noisy_ampa = two_type_pool(self.tau_damp_noisy_exc_ampa, self.tau_damp_noisy_inh_ampa)
    self.amp_noisy_gaba1 = two_type_pool(self.amp_noisy_exc_gaba1, self.amp_noisy_inh_gaba1)
    self.firing_rate_noisy_gaba1 = two_type_pool(self.firing_rate_noisy_exc_gaba1, self.firing_rate_noisy_inh_gaba1)
    self.tau_rise_noisy_gaba1 = two_type_pool(self.tau_rise_noisy_exc_gaba1, self.tau_rise_noisy_inh_gaba1)
    self.tau_damp_noisy_gaba1 = two_type_pool(self.tau_damp_noisy_exc_gaba1, self.tau_damp_noisy_inh_gaba1)
    self.amp_noisy_gaba2 = two_type_pool(self.amp_noisy_exc_gaba2, self.amp_noisy_inh_gaba2)
    self.firing_rate_noisy_gaba2 = two_type_pool(self.firing_rate_noisy_exc_gaba2, self.firing_rate_noisy_inh_gaba2)
    self.tau_rise_noisy_gaba2 = two_type_pool(self.tau_rise_noisy_exc_gaba2, self.tau_rise_noisy_inh_gaba2)
    self.tau_damp_noisy_gaba2 = two_type_pool(self.tau_damp_noisy_exc_gaba2, self.tau_damp_noisy_inh_gaba2)

    self.lgn_on_pos = DirectConnection(tau_rise = self.tau_rise_lgn_on_pos, tau_damp = self.tau_damp_lgn_on_pos, shape = self.shape, stimulus = self.stimulus, kernels = self.lgn_kernels_on, ikernels = self.lgn_ikernels_on)
    self.lgn_on_neg = DirectConnection(tau_rise = self.tau_rise_lgn_on_neg, tau_damp = self.tau_damp_lgn_on_neg, shape = self.shape, stimulus = self.stimulus, kernels = self.lgn_kernels_on, ikernels = self.lgn_ikernels_on)
    self.lgn_off_pos = DirectConnection(tau_rise = self.tau_rise_lgn_off_pos, tau_damp = self.tau_damp_lgn_off_pos, shape = self.shape, stimulus = self.stimulus, kernels = self.lgn_kernels_off, ikernels = self.lgn_ikernels_off)
    self.lgn_off_neg = DirectConnection(tau_rise = self.tau_rise_lgn_off_neg, tau_damp = self.tau_damp_lgn_off_neg, shape = self.shape, stimulus = self.stimulus, kernels = self.lgn_kernels_off, ikernels = self.lgn_ikernels_off)
    self.noisy_nmda = NoisyConnection(tau_rise_pool = self.tau_rise_noisy_nmda, tau_damp_pool = self.tau_damp_noisy_nmda, shape = self.shape, firing_rate_pool = self.firing_rate_noisy_nmda, amp_pool = self.amp_noisy_nmda)
    self.noisy_ampa = NoisyConnection(tau_rise_pool = self.tau_rise_noisy_ampa, tau_damp_pool = self.tau_damp_noisy_ampa, shape = self.shape, firing_rate_pool = self.firing_rate_noisy_ampa, amp_pool = self.amp_noisy_ampa)
    self.noisy_gaba1 = NoisyConnection(tau_rise_pool = self.tau_rise_noisy_gaba1, tau_damp_pool = self.tau_damp_noisy_gaba1, shape = self.shape, firing_rate_pool = self.firing_rate_noisy_gaba1, amp_pool = self.amp_noisy_gaba1)
    self.noisy_gaba2 = NoisyConnection(tau_rise_pool = self.tau_rise_noisy_gaba2, tau_damp_pool = self.tau_damp_noisy_gaba2, shape = self.shape, firing_rate_pool = self.firing_rate_noisy_gaba2, amp_pool = self.amp_noisy_gaba2)

  def step(self, t, dt):
    NeuronGroup.step(self, t, dt)
    rk2params(self.lgn_on_pos.g, self.lgn_on_neg.g, self.lgn_off_pos.g, self.lgn_off_neg.g, self.noisy_gaba1, self.noisy_gaba2, self.noisy_nmda, self.noisy_ampa, self.fgaba_noise, slef.fnmda_noise, self.alpha0, self.beta0)
    self.lgn_on_pos.step(t, dt)
    self.lgn_on_neg.step(t, dt)
    self.lgn_off_pos.step(t, dt)
    self.lgn_off_neg.step(t, dt)
    self.noisy_nmda.step(t, dt)
    self.noisy_gaba1.step(t, dt)
    self.noisy_ampa.step(t, dt)
    self.noisy_gaba2.step(t, dt)
    rk2params(self.lgn_on_pos.g, self.lgn_on_neg.g, self.lgn_off_pos.g, self.lgn_off_neg.g, self.noisy_gaba1, self.noisy_gaba2, self.noisy_nmda, self.noisy_ampa, self.fgaba_noise, slef.fnmda_noise, self.alpha1, self.beta1)
    rk2voltage(self.v, self.tspikes, self.trefs, self.alpha0, self.beta0, self.alpha1, self.beta1, self.v_thre, self.v_reset, t, dt)


