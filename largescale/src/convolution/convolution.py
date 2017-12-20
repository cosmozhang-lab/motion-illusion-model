import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "convolution.cl") )

class Conv2DKernel:
  def __init__(self, init):
    if isinstance(init, Conv2DKernel):
      self.kernel = init.kernel
      self.kernel_dev = init.kernel_dev
    elif isinstance(init, np.ndarray):
      self.kernel = init.astype(np.double)
      self.kernel_dev = clspt.Variable(self.kernel, read_only = True)
    else:
      raise TypeError("Initial value must be either a Conv2DKernel or a ndarray")
    self.shape = self.kernel.shape
    self.kernel_shape = self.kernel.shape
    self.kernel_shape_dev = clspt.Variable(np.array(self.kernel_shape), read_only = True)

class Conv2DKernelPool:
  def __init__(self, kernels):
    self.kernels = []
    self.shapes = []
    for kernel in kernels:
      if isinstance(kernel, Conv2DKernel):
        self.kernels.append(kernel.kernel)
        self.shapes.append(kernel.kernel_shape)
      elif isinstance(kernel, np.ndarray):
        self.kernels.append(kernel.astype(np.double))
        self.shapes.append(kernel.shape)
      else:
        raise TypeError("Kernel must be either a Conv2DKernel or a ndarray")
    self.kernels_host = np.zeros( (np.sum([np.prod(shape) for shape in self.shapes]),) ).astype(np.double)
    self.shapes_host = np.zeros( (2*len(self.shapes),) ).astype(np.int32)
    i = 0
    for kernel in self.kernels:
      kernel = kernel.ravel().astype(np.double)
      self.kernels_host[i:(i+kernel.size)] = kernel
      i = i + kernel.size
    i = 0
    for shape in self.shapes:
      self.shapes_host[i*2] = shape[0]
      self.shapes_host[i*2+1] = shape[1]
      i += 1
    self.kernels_dev = clspt.Variable( self.kernels_host, read_only = True )
    self.shapes_dev = clspt.Variable( self.shapes_host, read_only = True )

# 2D Convolution.
# Using "same" padding principle.
# The kernel to use for each pixel is one of the kernels in `kernel_pool`,
# and the index of the kernel to use for each function is specified in 
# `ikernels`. So input_map, output_map and ikernels should have the same 
# shape.
# @param input_map:   [Variable]<double>
# @param kernel_pool: [Conv2DKernelPool]
# @param ikernels:    [Variable]<int32>
# @param output_pam:  [Variable]<double>
# @kwarg queue:       [CommandQueue]
# @kwarg update:      [Boolean] whether to update the output_map
kernelf_conv2d = program.conv2d
def conv2d(input_map, kernel_pool, ikernels, output_map, queue = None, update=True):
  if queue is None:
    queue = clspt.queue()
  if not (input_map.shape == output_map.shape and input_map.shape == ikernels.shape):
    raise ValueError("The size of input, output and ikernels must be equal")
  if isinstance(ikernels, np.ndarray):
    ikernels = clspt.Variable(ikernels, read_only = True)
  rows = input_map.shape[0]
  cols = input_map.shape[1]
  nthreads = input_map.shape[0] * input_map.shape[1]
  kernelf_conv2d(queue, (nthreads,), None, rows, cols, input_map.buf_dev, kernel_pool.kernels_dev.buf_dev, kernel_pool.shapes_dev.buf_dev, ikernels.buf_dev, output_map.swp_dev)
  if update:
    output_map.update(queue)
