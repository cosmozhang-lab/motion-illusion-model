import pyopencl as cl

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

class Variable:
  def __init__(self, init=None, nbytes=None, read_only=False):
    ctx = context()
    self.readonly = read_only
    self.swappable = not read_only
    self.buf_host = init
    mode = mf.READ_ONLY if read_only else mf.READ_WRITE
    if not init is None:
      self.buf_dev = cl.Buffer(ctx, mode | mf.COPY_HOST_PTR, hostbuf = init)
      if self.swappable:
        self.swp_dev = cl.Buffer(ctx, mode | mf.COPY_HOST_PTR, hostbuf = init)
      else:
        self.swp_dev = None
    elif not nbytes is None:
      self.buf_dev = cl.Buffer(ctx, mode, nbytes)
      if self.swappable:
        self.swp_dev = cl.Buffer(ctx, mode, nbytes)
      else:
        self.swp_dev = None
    else:
      raise "Cannot create variable: must set the init or nbytes"

  def update(self):
    if self.swappable:
      tmp = self.swp_dev
      self.swp_dev = self.buf_dev
      self.buf_dev = tmp
