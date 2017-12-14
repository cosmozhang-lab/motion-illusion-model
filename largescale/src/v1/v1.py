# V1 neurons

from largescale.src.neuron import NeuronGroup
import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import largescale.src.neuron.program as program

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
  #     v_lgn: the resting voltage for LGN inputs
  #     v_exc: the resting voltage for excitatory
  #     v_inh: the resting voltage for inhibitory
  #     tau_rise_gaba1: the time constant of conductance rising for gaba1
  #     tau_damp_gaba1: the time constant of conductance damping for gaba1
  #     tau_rise_gaba2: the time constant of conductance rising for gaba2
  #     tau_damp_gaba2: the time constant of conductance damping for gaba2
  #     tau_rise_ampa: the time constant of conductance rising for ampa
  #     tau_damp_ampa: the time constant of conductance damping for ampa
  #     tau_rise_nmda: the time constant of conductance rising for nmda
  #     tau_damp_nmda: the time constant of conductance damping for nmda
  def __init__(self, nshape, config=None):
    NeuronGroup.__init__(self, nshape, config)
    self.type = None
    self.v_l = 0.0
    self.v_exc = 0.0
    self.v_inh = 0.0
    self.tau_rise_gaba1 = 0.0
    self.tau_damp_gaba1 = 0.0
    self.tau_rise_gaba2 = 0.0
    self.tau_damp_gaba2 = 0.0
    self.tau_rise_ampa = 0.0
    self.tau_damp_ampa = 0.0
    self.tau_rise_nmda = 0.0
    self.tau_damp_nmda = 0.0
    if "type" in config: self.type = clspt.Variable( np.array(config["type"]).astype(np.uint8), read_only = True )
    if "v_l" in config: self.v_l = config["v_l"]
    if "v_exc" in config: self.v_exc = config["v_exc"]
    if "v_inh" in config: self.v_inh = config["v_inh"]
    if "tau_rise_gaba1" in config: self.tau_rise_gaba1 = config["tau_rise_gaba1"]
    if "tau_damp_gaba1" in config: self.tau_damp_gaba1 = config["tau_damp_gaba1"]
    if "tau_rise_gaba2" in config: self.tau_rise_gaba2 = config["tau_rise_gaba2"]
    if "tau_damp_gaba2" in config: self.tau_damp_gaba2 = config["tau_damp_gaba2"]
    if "tau_rise_ampa" in config: self.tau_rise_ampa = config["tau_rise_ampa"]
    if "tau_damp_ampa" in config: self.tau_damp_ampa = config["tau_damp_ampa"]
    if "tau_rise_nmda" in config: self.tau_rise_nmda = config["tau_rise_nmda"]
    if "tau_damp_nmda" in config: self.tau_damp_nmda = config["tau_damp_nmda"]
    self.g_lgn = clspt.Variable( np.zeros(nshape).astype(np.double) )
    self.s_lgn = clspt.Variable( np.zeros(nshape).astype(np.double) )
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
    program.chain2(queue, (self.nneurons,), None, self.g_gaba1.buf_dev, self.s_gaba1.buf_dev, self.tau_rise_gaba1, self.tau_damp_gaba1, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_gaba2.buf_dev, self.s_gaba2.buf_dev, self.tau_rise_gaba2, self.tau_damp_gaba2, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_ampa.buf_dev, self.s_ampa.buf_dev, self.tau_rise_ampa, self.tau_damp_ampa, dt)
    program.chain2(queue, (self.nneurons,), None, self.g_nmda.buf_dev, self.s_nmda.buf_dev, self.tau_rise_nmda, self.tau_damp_nmda, dt)
