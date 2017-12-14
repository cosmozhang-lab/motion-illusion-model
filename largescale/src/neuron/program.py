import pyopencl as cl
import largescale.src.support.cl_support as clspt
import numpy as np
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]

program_file = open( os.path.join(thisdir, "program.cl") )
program = cl.Program(clspt.context(), program_file.read()).build()
program_file.close()

chain2 = program.chain2
chain2.set_scalar_arg_dtypes([None, None, np.double, np.double, np.double])
