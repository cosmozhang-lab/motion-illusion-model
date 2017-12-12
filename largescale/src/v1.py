# V1 neurons

from largescale.src.neuron import NeuronBase

class V1Neuron (NeuronBase):
  def __init__(self, lgn_neurons):
    self.lgn = lgn_neurons
    NeuronBase.__init__(self)

  def step(self, dt):
    NeuronBase.step(self, dt)



if __name__ == "__main__":
  n = V1Neuron(None)
  print n.time
  n.step(0.01)
  print n.time