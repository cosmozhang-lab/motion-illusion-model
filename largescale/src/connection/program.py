import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support import CommonConfig
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "program.cl") )

kernel_chain2_with_input = program.chain2_with_input.kernel
kernel_input = program.input.kernel
