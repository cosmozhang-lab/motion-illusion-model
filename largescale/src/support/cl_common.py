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
# @kwarg update: [Boolean] whether to update the result variable immediately (default is True)
kernel_add = program.add
def add(*args, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = True
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
  return result

# @Function sub([CommandQueue queue,] Variable y, Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param y:      [Variable]<double> subtrahend variable
# @param xi:     [Variable]<double> subtracter variable
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is True)
kernel_sub = program.sub
def sub(*args, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = True
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
  return result

# @Function dotmul([CommandQueue queue,] Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param queue:  [CommandQueue]
# @param xi:     [Variable]<double> multiplying variable
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is True)
kernel_dotmul = program.dotmul
def dotmul(*args, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = True
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
    kernel_dotmul(queue, (result.size,), None, x0, x1, result.buf_swp)
    x0 = result.buf_swp
  if with_update:
    result.update(queue)
  return result

# @Function timed([CommandQueue queue,] Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param queue:  [CommandQueue]
# @param c:      [double] timing coefficient
# @param x:      [Variable]<double> timing variable
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is True)
kernel_timed = program.timed
def timed(c, x, result, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = True
  if "queue" in kwargs:
    queue = kwargs["queue"]
  else:
    queue = clspt.queue()
  if not x.size == result.size:
    raise ValueError("Variable sizes must be equal")
  kernel_timed(queue, (result.size,), None, c, x.buf_dev, result.buf_swp)
  if with_update:
    result.update(queue)
  return result

# @Function inverse([CommandQueue queue,] Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param queue:  [CommandQueue]
# @param x:      [Variable]<double> variable to be inversed
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is True)
kernel_inverse = program.inverse
def inverse(c, x, result, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = True
  if "queue" in kwargs:
    queue = kwargs["queue"]
  else:
    queue = clspt.queue()
  if not x.size == result.size:
    raise ValueError("Variable sizes must be equal")
  kernel_inverse(queue, (result.size,), None, x.buf_dev, result.buf_swp)
  if with_update:
    result.update(queue)
  return result

# @Function minus([CommandQueue queue,] Variable x1, Variable x2, ..., Variable xn, Variable result, **kwargs)
# @param queue:  [CommandQueue]
# @param x:      [Variable]<double> variable to be minused
# @param result: [Variable]<double> storing the result
# @kwarg queue:  [CommandQueue]
# @kwarg update: [Boolean] whether to update the result variable immediately (default is True)
kernel_minus = program.minus
def minus(c, x, result, **kwargs):
  if "update" in kwargs:
    with_update = kwargs["update"]
  else:
    with_update = True
  if "queue" in kwargs:
    queue = kwargs["queue"]
  else:
    queue = clspt.queue()
  if not x.size == result.size:
    raise ValueError("Variable sizes must be equal")
  kernel_minus(queue, (result.size,), None, x.buf_dev, result.buf_swp)
  if with_update:
    result.update(queue)
  return result

