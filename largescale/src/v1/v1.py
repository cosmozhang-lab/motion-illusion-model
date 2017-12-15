# V1 neurons

from largescale.src.neuron import NeuronGroup
import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import largescale.src.neuron.program as program
from largescale.src.support import CommonConfig

T_EXCITATORY = 1 # "V1NeuronType_Excitatory"
T_INHIBITORY = 2 # "V1NeuronType_Inhibitory"
T_EXC = T_EXCITATORY
T_E = T_EXCITATORY
T_INH = T_INHIBITORY
T_I = T_INHIBITORY

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
    self.type = clspt.Variable( np.array(config.type).astype(np.uint8), read_only = True ) if config.type else None
    self.stimulus = config.fetch("stimulus")
    self.v_l = config.fetch("v_l", 0.0)
    self.v_exc = config.fetch("v_exc", 0.0)
    self.v_inh = config.fetch("v_inh", 0.0)
    self.tau_rise_lgn_on = config.fetch("tau_rise_lgn_on", 0.0)
    self.tau_damp_lgn_on = config.fetch("tau_damp_lgn_on", 0.0)
    self.tau_rise_lgn_off = config.fetch("tau_rise_lgn_off", 0.0)
    self.tau_damp_lgn_off = config.fetch("tau_damp_lgn_off", 0.0)
    self.tau_rise_gaba1 = config.fetch("tau_rise_gaba1", 0.0)
    self.tau_damp_gaba1 = config.fetch("tau_damp_gaba1", 0.0)
    self.tau_rise_gaba2 = config.fetch("tau_rise_gaba2", 0.0)
    self.tau_damp_gaba2 = config.fetch("tau_damp_gaba2", 0.0)
    self.tau_rise_ampa = config.fetch("tau_rise_ampa", 0.0)
    self.tau_damp_ampa = config.fetch("tau_damp_ampa", 0.0)
    self.tau_rise_nmda = config.fetch("tau_rise_nmda", 0.0)
    self.tau_damp_nmda = config.fetch("tau_damp_nmda", 0.0)
    self.g_leak = config.fetch("g_leak", 0.0)
    self.kernels = config.fetch("kernels", None)
    self.ikernel = clspt.Variable( config.fetch("ikernel", np.zeros(nshape).astype(np.double)), read_only = True )
    self.g_lgn_on = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_lgn_on = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.g_lgn_off = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_lgn_off = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.g_ampa = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_ampa = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.g_nmda = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_nmda = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.g_gaba1 = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_gaba1 = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.g_gaba2 = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_gaba2 = clspt.Variable( np.zeros(nshape).astype(np.double) )

  def step(self, t, dt):
    NeuronGroup.step(self, t, dt)
    queue = clspt.queue()
    program.chain2(queue, (self.nneurons,), None, self.g_lgn_on.buf_dev, self.s_lgn_on.buf_dev, self.tau_rise_lgn_on, self.tau_damp_lgn_on, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_lgn_off.buf_dev, self.s_lgn_off.buf_dev, self.tau_rise_lgn_off, self.tau_damp_lgn_off, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_gaba1.buf_dev, self.s_gaba1.buf_dev, self.tau_rise_gaba1, self.tau_damp_gaba1, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_gaba2.buf_dev, self.s_gaba2.buf_dev, self.tau_rise_gaba2, self.tau_damp_gaba2, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_ampa.buf_dev, self.s_ampa.buf_dev, self.tau_rise_ampa, self.tau_damp_ampa, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_nmda.buf_dev, self.s_nmda.buf_dev, self.tau_rise_nmda, self.tau_damp_nmda, dt)

  def update(self):
    self.g_gaba1.update()
    self.s_gaba1.update()
    self.g_gaba2.update()
    self.s_gaba2.update()
    self.g_ampa.update()
    self.s_ampa.update()
    self.g_nmda.update()
    self.s_nmda.update()
