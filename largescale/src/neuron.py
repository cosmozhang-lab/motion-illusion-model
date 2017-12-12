# Neuron base

class NeuronBase:
  def __init__(self):
    self.initialize()

  def initialize(self):
    self.time = 0.0

  def step(self, dt):
    self.time += dt

