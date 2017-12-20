import pyopencl as cl
import cl_support as clspt
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "cl_common.cl") )

# @Function add([CommandQueue queue,] Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param queue:  [CommandQueue]
# @param xi:     [Variable]<double> adding variable
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is False)
kernel_add = program.add
def add(*args, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = False
  if "queue" in kwargs:
    queue = kwargs["queue"]
  else:
    queue = clspt.queue()
  result = args[-1]
  x = args[0:-1]
  for xx in x:
    if not xx.size == result.size:
      raise ValueError("Variable sizes must be equal")
  x0 = x[0].buf_dev
  for i in xrange(1, len(x)):
    x1 = x[i].buf_dev
    kernel_add(queue, (result.size,), None, x0, x1, result.buf_swp)
    x0 = result.buf_swp
  if with_update:
    result.update(queue)

# @Function sub([CommandQueue queue,] Variable y, Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param y:      [Variable]<double> subtrahend variable
# @param xi:     [Variable]<double> subtracter variable
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is False)
kernel_sub = program.sub
def sub(*args, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = False
  if "queue" in kwargs:
    queue = kwargs["queue"]
  else:
    queue = clspt.queue()
  result = args[-1]
  x = args[0:-1]
  for xx in x:
    if not xx.size == result.size:
      raise ValueError("Variable sizes must be equal")
  x0 = x[0].buf_dev
  for i in xrange(1, len(x)):
    x1 = x[i].buf_dev
    kernel_add(queue, (result.size,), None, x0, x1, result.buf_swp)
    x0 = result.buf_swp
  if with_update:
    result.update(queue)

