import pyopencl as cl
import numpy as np
import re
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]

mem_flags = cl.mem_flags
mf = mem_flags

_context = None

def get_context():
  global _context
  if _context is None:
    devices = []
    for platform in cl.get_platforms():
      devices += platform.get_devices()
    _context = cl.Context(devices)
  return _context
context = get_context

_queues = {}
default_queue_name = "default"
def get_queue(name=None):
  if name is None: name = default_queue_name
  if not name in _queues:
    ctx = context()
    queue = cl.CommandQueue(ctx)
    _queues[name] = queue
  return _queues[name]
queue = get_queue

class CLKernelArgMeta:
  def __init__(self, name, **kwargs):
    if "typestr" in kwargs:
      def word2type(word):
        if word == "double":
          return np.float64
        elif word == "float":
          return np.float32
        elif word == "int":
          return np.int32
        elif word == "unsigned int":
          return np.uint32
        elif word == "long":
          return np.int32
        elif word == "unsigned long":
          return np.uint32
        elif word == "char":
          return np.int8
        elif word == "unsigned char":
          return np.uint8
        elif word == "long long":
          return np.int64
        elif word == "unsigned long long":
          return np.uint64
        else:
          return None
      typestr = kwargs["typestr"]
      regexp = re.compile(r"\W")
      argwords = []
      last_idx = 0
      for match in regexp.finditer(typestr):
        match_start = match.start()
        match_text = match.group(0)
        if last_idx < match_start:
          argwords.append(typestr[last_idx:match_start])
        if match_text == "*":
          argwords.append(match_text)
        last_idx = match.end()
      if last_idx < len(typestr):
        argwords.append(typestr[last_idx:])
      self.islocal = "__local" in argwords
      self.isglobal = "__global" in argwords
      self.isconst = "const" in argwords
      self.dtype = None
      typeword = ""
      typewords = ["unsigned", "int", "double", "float" "long"]
      for word in argwords:
        if word in typewords:
          typeword += " " + word
        elif len(typeword) > 0:
          self.dtype = word2type(typeword.strip())
          typeword = ""
      if len(typeword) > 0:
        self.dtype = word2type(typeword.strip())
    else:
      self.isglobal = False
      self.islocal = False
      self.isconst = False
      self.dtype = None
    self.isglobal = kwargs["isglobal"] if "isglobal" in kwargs else self.isglobal
    self.islocal = kwargs["islocal"] if "islocal" in kwargs else self.islocal
    self.isconst = kwargs["isconst"] if "isconst" in kwargs else self.isconst
    self.dtype = kwargs["dtype"] if "dtype" in kwargs else self.dtype
  
  @property
  def isbuffer(self):
    return self.islocal or self.isglobal

  @property
  def scalar_type(self):
    if self.isbuffer:
      return None
    else:
      return self.dtype

class CLKernel:
  def __init__(self, kernel, arg_metas = []):
    self.kernel = kernel
    self.arg_metas = arg_metas
    self.kernel.set_scalar_arg_dtypes([arg.scalar_type for arg in arg_metas])
  def __call__(self, *args, **kwargs):
    size = args[0]
    args = args[1:]
    if not isinstance(size, tuple) and not isinstance(size, list):
      size = (size,)
    queue = kwargs["queue"] if queue in kwargs else get_queue()
    update = kwargs["update"] if update in kwargs else get_queue()
    kargs = [queue, size, None]
    for i in xrange(len(args)):
      arg = args[i]
      meta = self.arg_metas[i]
      if isinstance(arg, Variable):
        if meta.isconst:
          arg = arg.buf_dev
        else:
          arg = arg.swp_dev
      kargs.append(arg)
    apply(self.kernel, kargs)
    if update:
      for i in xrange(len(args)):
        arg = args[i]
        meta = self.arg_metas[i]
        if isinstance(arg, Variable) and not meta.isconst:
          arg.update(queue)

# compile a cl program
class CLProgram:
  def __init__(self, filename = None, code = None):
    self.kernel_dict = {}
    ctx = context()
    filedir = None
    assert (filename is None) or (code is None), "Missing program code / file"
    if code is None:
      file = open(filename)
      code = file.read()
      file.close()
      filedir = os.path.split(os.path.realpath(filename))[0]
    # prepare
    global_dir = os.path.realpath(os.path.join(thisdir, ".."))
    lines = code.split("\n")
    # process includes
    re_include_global = re.compile(r"^\s*#include\s+\<([^\<^\>]+)\>\s*\;?\s*$")
    re_include_local = re.compile(r"^\s*#include\s+\"([^\"]+)\"\s*\;?\s*$")
    while True:
      newincluded = False
      newlines = []
      for i in xrange(len(lines)):
        line = lines[i]
        incfilename = None
        match = re_include_global.match(line)
        if match:
          incfilename = os.path.join(global_dir, match.group(1))
        elif not filedir is None:
          match = re_include_local.match(line)
          if match:
            incfilename = os.path.join(filedir, match.group(1))
        if not incfilename is None:
          incfile = open(incfilename)
          newlines.extend(incfile.read().split("\n"))
          line = None
          newincluded = True
          incfile.close()
        if not line is None:
          newlines.append(line)
      lines = newlines
      if not newincluded:
        break
    # replace the comments
    re_comment = re.compile(r"\/\/.*$")
    for i in xrange(len(lines)):
      lines[i] = re_comment.sub("", lines[i])
    code = "".join(lines)
    in_comment = False
    comment_pos = 0
    comments = []
    for i in xrange(len(code)):
      if not in_comment and code[i:i+2] == "/*":
        in_comment = True
        comment_pos = i
      elif in_comment and code[i:i+2] == "*/":
        in_comment = False
        comments.append((comment_pos,i+2))
    if len(comments) > 0:
      code = code[0:comments[0][0]] + "".join([code[comments[i][1]:comments[i+1][0]] for i in xrange(len(comments)-1)]) + code[comments[-1][1]:]
    # extract function metas
    metas = {}
    re_function_header = re.compile(r"__kernel\s+void\s+(\w+)\s*\(([^\(\)]+)\)")
    re_function_arg = re.compile(r"(^[\w\s]+)([\*\s]\s*)(\w+)$")
    # re_function_argtype_global_mem = re.compile(r"^__global\s+(const\s+)?[\w\s]+\*$")
    # re_function_argtype_local_mem = re.compile(r"^__local\s+(const\s+)?[\w\s]+\*$")
    for match in re_function_header.finditer(code):
      fn_name = match.group(1)
      fn_args = match.group(2).split(",")
      arg_metas = []
      for arg in fn_args:
        arg_match = re_function_arg.match(arg.strip())
        arg_typestr = arg_match.group(1) + arg_match.group(2)
        arg_name = arg_match.group(3)
        arg_metas.append(CLKernelArgMeta(arg_name, typestr=arg_typestr))
      metas[fn_name] = {"args": arg_metas}
    # compile
    self.program = cl.Program(ctx, code).build()
    # extract functions and attach arguments
    kernels = self.program.all_kernels()
    for kernel in kernels:
      fn_name = kernel.get_info(cl.kernel_info.FUNCTION_NAME)
      if not fn_name in metas: continue
      meta = metas[fn_name]
      self.kernel_dict[fn_name] = CLKernel(kernel, meta["args"])

  def __getattr__(self, name):
    if name in self.kernel_dict:
      return self.kernel_dict[name]
    return object.__getattr__(self, name)

# Compile an OpenCL program
def compile(filename = None, code = None):
  return CLProgram(filename = filename, code = code)



class Variable:
  def __init__(self, init=None, shape=None, dtype=None, read_only=False, auto_update=False):
    ctx = context()
    self.readonly = read_only
    self.swappable = (not read_only) or (not auto_update)
    mode = mf.READ_ONLY if read_only else mf.READ_WRITE
    if not init is None:
      self._buf_host = init
      self.shape = self._buf_host.shape
      self.dtype = self._buf_host.dtype
      self._buf_dev = cl.Buffer(ctx, mode | mf.COPY_HOST_PTR, hostbuf = init)
      if self.swappable:
        self._swp_dev = cl.Buffer(ctx, mode | mf.COPY_HOST_PTR, hostbuf = init)
      else:
        self._swp_dev = None
    elif not shape is None and not dtype is None:
      self.shape = shape
      self.dtype = dtype
      self._buf_host = np.zeros(self.shape).astype(self.dtype)
      self._buf_dev = cl.Buffer(ctx, mode, self._buf_host.nbytes)
      if self.swappable:
        self._swp_dev = cl.Buffer(ctx, mode, self._buf_host.nbytes)
      else:
        self._swp_dev = None
    else:
      raise "Cannot create variable: must set the initial value or shape&dtype"
    self.dirty = False

  def __del__(self):
    if self._buf_dev:
      self._buf_dev.release()
      self._buf_dev = None
    if self._swp_dev:
      self._swp_dev.release()
      self._swp_dev = None

  @property
  def buf_dev(self):
    return self._buf_dev
  @property
  def swp_dev(self):
    if self.swappable:
      return self._swp_dev
    else:
      return self._buf_dev
  @property
  def buf_host(self):
    return self.fetch()

  @property
  def nbytes(self):
    return self._buf_host.nbytes

  @property
  def size(self):
    return self._buf_host.size

  def update(self, queue = None):
    if self.swappable:
      cl.enqueue_barrier(queue or get_queue())
      tmp = self._swp_dev
      self._swp_dev = self._buf_dev
      self._buf_dev = tmp
    if not self.readonly:
      self.dirty = True

  def fetch(self, queue = None):
    cl.enqueue_copy(queue or get_queue(), self._buf_host, self._buf_dev)
    self.dirty = False
    return self._buf_host

  def fill(self, src, queue = None):
    nbytes = self._buf_host.nbytes
    if isinstance(src, np.ndarray):
      nbytes = src.nbytes
    else:
      src = np.array(src).astyle(self.dtype)
    cl.enqueue_fill_buffer(queue or get_queue(), self._buf_dev, src, 0, nbytes)
    self.dirty = True


# Compile a expression
def map_kernel(expr, name = None):
  import re
  class ParamItem:
    def __init__(self, match=None, isout=False, name=None):
      if isout:
        self.name = name
        self.isbuf = True
        self.isout = True
      else:
        self.name = match.group(1)
        self.isbuf = not match.group(2) is None
        self.isout = False
    def toString(self):
      if self.isbuf:
        if self.isout:
          return "__global float * " + self.name
        else:
          return "__global const float * " + self.name
      else:
        return "float " + self.name
  if name is None: name = "_map_kernel_function" # the default name
  result_param_name = "_map_kernel_function_result" # name of the output buffer argument
  params = []
  param_names = []
  for m in re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(\[i\])?").finditer(expr):
    p = ParamItem(m)
    if not p.name in param_names:
      params.append(p)
      param_names.append(p.name)
  params.append(ParamItem(isout=True, name=result_param_name))
  paramstr = ", ".join([p.toString() for p in params])
  code = "__kernel void %s (%s) { int i = get_global_id(0); %s[i] = %s; }" % (
      name, paramstr, result_param_name, expr )
  kernel = getattr(compile(code = code), name)
  class MapKernel:
    def __init__(self, kernel, params):
      self.kernel = kernel
      self.params = params
      self.param_names = [p.name for p in self.params]
    def __call__(self, *args, **kwargs):
      params = [None for p in self.params]
      nargs = len(args)
      for i in xrange(nargs):
        params[i] = args[i]
      for i in xrange(nargs, len(params)):
        k = self.param_names[i]
        if k in kwargs:
          params[i] = kwargs[k]
        else:
          if i < len(params) - 1:
            raise TypeError("Argument [%s](#%d) is not given" % (k,i))
      if "out" in kwargs: params[-1] = kwargs["out"]
      if "result" in kwargs: params[-1] = kwargs["result"]
      if "output" in kwargs: params[-1] = kwargs["output"]
      outsize = params[-1].size
      for i in xrange(len(params)):
        if self.params[i].isbuf and not isinstance(params[i], Variable):
          raise TypeError("Argument [%s](#%d) should be a buffer" % (self.params[i].name, i))
        elif not self.params[i].isbuf and isinstance(params[i], Variable):
          raise TypeError("Argument [%s](#%d) should be a buffer" % (self.params[i].name, i))
        if self.params[i].isbuf and not (params[i].size == outsize):
          raise ValueError("Size of argument [%s](#%d) is not compatible with output buffer" % (self.params[i].name, i))
      queue = kwargs["queue"] if "queue" in kwargs else get_queue()
      update = kwargs["update"] if "update" in kwargs else True
      inparams = [queue, (params[-1].size,), None]
      for i in xrange(len(params)):
        if self.params[i].isbuf:
          if self.params[i].isout:
            inparams.append(params[i].swp_dev)
          else:
            inparams.append(params[i].buf_dev)
        else:
          inparams.append(params[i])
      apply(self.kernel, inparams)
      if update:
        for i in xrange(len(params)):
          if self.params[i].isbuf and self.params[i].isout:
            params[i].update(queue)
  return MapKernel(kernel, params)



RAND_MAX = 2**32-1
