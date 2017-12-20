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
    re_function_arg = re.compile(r"(^[\w\s]+)([\*\s]\s*)(\w+)\s*$")
    re_function_argtype_global_mem = re.compile(r"^__global\s+[\w\s]+\*$")
    re_function_argtype_local_mem = re.compile(r"^__local\s+[\w\s]+\*$")
    for match in re_function_header.finditer(code):
      fn_name = match.group(1)
      fn_args = match.group(2).split(",")
      arg_metas = []
      for arg in fn_args:
        match = re_function_arg.match(arg)
        arg_name = match.group(3)
        arg_typestr = (match.group(1) + match.group(2)).strip()
        arg_type = None
        if re_function_argtype_global_mem.match(arg_typestr):
          arg_type = None
        elif re_function_argtype_local_mem.match(arg_typestr):
          arg_type = None
        elif arg_typestr == "double":
          arg_type = np.double
        elif arg_typestr == "int":
          arg_type = np.int32
        elif arg_typestr == "unsigned int":
          arg_type = np.uint32
        elif arg_typestr == "long":
          arg_type = np.int32
        elif arg_typestr == "unsigned long":
          arg_type = np.uint32
        elif arg_typestr == "char":
          arg_type = np.int8
        elif arg_typestr == "unsigned char":
          arg_type = np.uint8
        elif arg_typestr == "long long":
          arg_type = np.int64
        elif arg_typestr == "unsigned long long":
          arg_type = np.uint64
        else:
          raise TypeError("Unknown argument type: %s" % arg_typestr)
        arg_metas.append({"name": arg_name, "type": arg_type})
      metas[fn_name] = {"args": arg_metas}
    # compile
    self.program = cl.Program(ctx, code).build()
    # extract functions and attach arguments
    kernels = self.program.all_kernels()
    for kernel in kernels:
      fn_name = kernel.get_info(cl.kernel_info.FUNCTION_NAME)
      if not fn_name in metas: continue
      meta = metas[fn_name]
      kernel.set_scalar_arg_dtypes([arg["type"] for arg in meta["args"]])
      self.kernel_dict[fn_name] = kernel

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

  def __del__(self):
    if self.buf_dev:
      self.buf_dev.release()
    if self.swp_dev:
      self.swp_dev.release()
    object.__del__(self)

  def update(self, queue = None):
    if self.swappable:
      cl.enqueue_barrier(queue or get_queue())
      tmp = self.swp_dev
      self.swp_dev = self.buf_dev
      self.buf_dev = tmp

  def fetch(self, queue = None):
    res = np.zeros(self.shape).astype(self.dtype)
    cl.enqueue_copy(queue or get_queue(), res, self.buf_dev)
    return res

  def fill(self, src, queue = None):
    cl.enqueue_fill_buffer(queue or get_queue(), self.buf_dev, src, 0, src.nbytes)

RAND_MAX = 2**32-1
