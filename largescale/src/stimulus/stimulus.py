# Visual stimuli

import numpy as np
from largescale.src.support import CommonConfig

# Stimulus is regarded as a source flow,
# which gives an visual image on any certain t.
# In this implementation, visual image should be
# an image in `size` and varies between -1 to 1
class StimulusBase:
  def __init__(self, size, config=CommonConfig()):
    self.size = size
    self.config = config
  def get(t):
    return np.zeros(self.size)
