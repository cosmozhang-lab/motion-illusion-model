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
    self.shape = self.size
    self.config = config
  # Get the stimulus at t
  # `var` is an optional device Variable. If this
  # is provided, the function will fill the `var`
  # with the new stimulus pattern. Else it will
  # return a `ndarray` containing the pattern.
  # @param var:  [Variable]<float> if given, fill this variable
  def get(self, t, var=None, queue = None):
    if var is None:
      return np.zeros(self.size)
    else:
      var.fill(0.0, queue = queue)
