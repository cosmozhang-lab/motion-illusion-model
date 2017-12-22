import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "program.cl") )

drifting_grating = program.drifting_grating