import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program_file = open( os.path.join(thisdir, "convolution.cl") )
program = cl.Program(clspt.context(), program_file.read()).build()
program_file.close()

kernelf_conv2d = program.conv2d

class Conv2DKernel:
  def __init__(self, init):
    if isinstance(init, Conv2DKernel):
      self.kernel = init.kernel
      self.kernel_dev = init.kernel_dev
    elif isinstance(init, np.ndarray):
      self.kernel = init
      self.kernel_dev = clspt.Variable(self.kernel, read_only = True)
    else:
      raise TypeError("Initial value must be either a Conv2DKernel or a ndarray")

class Conv2DKernelPool:
  def __init__(self, kernels):
    self.kernels = []
    for kernel in kernels:
      if isinstance(kernel, Conv2DKernel):
        self.kernels.append(kernel)
      elif isinstance(kernel, np.ndarray):
        self.kernels.append(Conv2DKernel(kernel))
      else:
        raise TypeError("Kernel must be either a Conv2DKernel or a ndarray")
    self.kernel_dev_ptrs = np.array([k.kernel_dev.int_ptr for k in self.kernels])

def conv2d(queue, global_size, local_size, kernel_pool, ikernels, result):
  if isinstance(ikernels, np.ndarray):
    ikernels = clspt.Variable(ikernels, read_only = True)
  return kernelf_conv2d(queue, global_size, local_size, kernel_pool.kernel_dev_ptrs, ikernels.buf_dev, result.buf_dev)
