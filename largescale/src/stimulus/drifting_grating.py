from stimulus import StimulusBase
import numpy as np
import largescale.src.support.geometry as geo
from largescale.src.support import CommonConfig

# Drifting-grating stimulus
class DFStimulus (StimulusBase):
  def __init__(self, size, config = CommonConfig()):
    StimulusBase.__init__(self, size, config)
    self.orientation = config.fetch("orientation", 0.0)
    self.frequency = config.fetch("frequency", 1.0)
    self.speed = config.fetch("speed", 1.0) # drifting speed (orthogonal to orientation)
    self.phase = config.fetch("phase", 0.0) # initial phase
  def get(self, t):
    # I = cos(y'')
    # where: y'' = 2pi * (y' - speed * t) * frequency - phase
    # where: y' = x*sin(o) + y*cos(o)
    coors = geo.gen_coordinates(self.size)
    coorx = coors[1] - self.size[1] * 0.5
    coory = coors[0] - self.size[0] * 0.5
    yy = coorx * np.sin(self.orientation) + coory * np.cos(self.orientation)
    yyy = 2 * np.pi * (yy - self.speed * t) * self.frequency - self.phase
    img = np.cos(yyy)
    img = img
    return img

