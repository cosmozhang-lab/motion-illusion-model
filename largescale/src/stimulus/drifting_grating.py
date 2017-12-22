from stimulus import StimulusBase
import numpy as np
import largescale.src.support.geometry as geo
from largescale.src.support import CommonConfig
import largescale.src.support.cl_support as clspt
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "drifting_grating.cl") )

# Drifting-grating stimulus
class DFStimulus (StimulusBase):
  def __init__(self, size, config = CommonConfig()):
    StimulusBase.__init__(self, size, config)
    self.orientation = config.fetch("orientation", 0.0)
    self.frequency = config.fetch("frequency", 1.0)
    self.speed = config.fetch("speed", 1.0) # drifting speed (orthogonal to orientation)
    self.phase = config.fetch("phase", 0.0) # initial phase
  def get(self, t, var = None, queue = None):
    # I = cos(y'')
    # where: y'' = 2pi * (y' - speed * t) * frequency - phase
    #            = 2pi * (y' * frequency) - (speed * t * frequency + phase)
    # where: y' = x*sin(o) + y*cos(o)
    cos_o = np.cos(self.orientation)
    sin_o = np.sin(self.orientation)
    phase = self.speed * t * self.frequency + self.phase
    frequency = self.frequency
    rows = self.size[0]
    cols = self.size[1]
    if var is None:
      coors = geo.gen_coordinates(self.size)
      coorx = coors[1] - self.cols * 0.5
      coory = coors[0] - self.rows * 0.5
      yy = coorx * sin_o + coory * cos_o
      yyy = 2 * np.pi * yy * frequency - phase
      img = np.cos(yyy)
      return img
    else:
      program.drifting_grating.kernel(queue or clspt.queue(), (rows*cols,), None, rows, cols, var.swp_dev, sin_o, cos_o, frequency, phase, np.pi)
      var.update(queue or clspt.queue())

