import pyopencl as cl
import numpy as np

mem_flags = cl.mem_flags
mf = mem_flags

_context = None

def context():
  global _context
  if _context is None:
    devices = []
    for platform in cl.get_platforms():
      devices += platform.get_devices()
    _context = cl.Context(devices)
  return _context

_queues = {}
default_queue_name = "default"
def queue(name=None):
  if name is None: name = default_queue_name
  if not name in _queues:
    ctx = context()
    queue = cl.CommandQueue(ctx)
    _queues[name] = queue
  return _queues[name]

# Compile an OpenCL program
# def compile(filename = None, code = None):
#   ctx = context()
#   if code is None:
#     if filename is None:
#       raise "Missing program code / file"
#     else:
#       file = open(filename)
#       code = file.read()
#       file.close()
#   # preprocess the codes
#   meta = {}
#   lines = code.split("\n")
#   cl.Program(ctx, code)

class Variable:
  def __init__(self, init=None, shape=None, dtype=None, read_only=False):
    ctx = context()
    self.readonly = read_only
    self.swappable = not read_only
    mode = mf.READ_ONLY if read_only else mf.READ_WRITE
    if not init is None:
      self.buf_host = init
      self.shape = self.buf_host.shape
      self.dtype = self.buf_host.dtype
      self.buf_dev = cl.Buffer(ctx, mode | mf.COPY_HOST_PTR, hostbuf = init)
      if self.swappable:
        self.swp_dev = cl.Buffer(ctx, mode | mf.COPY_HOST_PTR, hostbuf = init)
      else:
        self.swp_dev = None
    elif not shape is None and not dtype is None:
      self.shape = shape
      self.dtype = dtype
      buf_host = np.zeros(self.shape).astype(self.dtype)
      self.buf_dev = cl.Buffer(ctx, mode, buf_host.nbytes)
      if self.swappable:
        self.swp_dev = cl.Buffer(ctx, mode, buf_host.nbytes)
      else:
        self.swp_dev = None
    else:
      raise "Cannot create variable: must set the initial value or shape&dtype"

  def update(self, command_queue = None):
    if self.swappable:
      cl.enqueue_barrier(command_queue or queue())
      tmp = self.swp_dev
      self.swp_dev = self.buf_dev
      self.buf_dev = tmp

  def fetch(self, command_queue = None):
    res = np.zeros(self.shape).astype(self.dtype)
    cl.enqueue_copy(command_queue or queue(), res, self.buf_dev)
    return res
