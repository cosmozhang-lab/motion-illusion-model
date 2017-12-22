# Unit testings

from largescale.src.support.common import CommonConfig

def test_v1():
  from largescale.src.v1 import V1DirectNeuronGroup
  import numpy as np
  config = CommonConfig({
    "tau_rise_gaba1": 0.1,
    "tau_damp_gaba1": 0.1,
    "tau_rise_gaba2": 0.1,
    "tau_damp_gaba2": 0.1,
    "tau_rise_ampa": 0.1,
    "tau_damp_ampa": 0.1,
    "tau_rise_nmda": 0.1,
    "tau_damp_nmda": 0.1
  })
  n = V1DirectNeuronGroup((288,144), config = config)
  import time
  t = time.time()
  dt = 0.001
  ts = np.arange(1000).astype(np.float32) * dt
  for tt in ts:
    n.step(tt, dt)
  print time.time() - t


def test_cl():
  import pyopencl as cl
  import largescale.src.support.cl_support as clspt
  import numpy as np
  import time
  n = 10
  ctx = clspt.context()
  prg = cl.Program(ctx, """
  inline float m_add(float a, float b) {
    return a + b;
  }
  __kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g)
  {
    int gid = get_global_id(0);
    res_g[gid] = m_add(a_g[gid], b_g[gid]); //a_g[gid] + b_g[gid];
  }
  """).build()
  mf = cl.mem_flags
  queues = []
  a_gs = []
  b_gs = []
  res_gs = []
  ni = 1
  for i in xrange(ni):
    # a_np = np.random.rand(n).astype(np.float32)
    # b_np = np.random.rand(n).astype(np.float32)
    a_np = np.arange(n).astype(np.float32)
    b_np = np.arange(n).astype(np.float32)
    a_gs.append( cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = a_np) )
    b_gs.append( cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = b_np) )
    res_gs.append( cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes) )
    queues.append( cl.CommandQueue(ctx) )
  t = time.time()
  for i in xrange(ni):
    prg.sum(queues[i], a_np.shape, None, a_gs[i], b_gs[i], res_gs[i])
    cl.enqueue_barrier(queues[i])
    print time.time() - t
  # cl.wait_for_events([cl.enqueue_marker(q) for q in queues])
  res_np = np.empty_like(a_np)
  cl.enqueue_copy(queues[0], res_np, res_gs[0])
  print res_np
  cl.enqueue_barrier(queues[-1])
  print time.time() - t

def test_geo():
  import largescale.src.support.geometry as geo
  import time
  t = time.time()
  coors = geo.gen_coordinates((288,144))
  print coors[0][:,0]
  print coors[1][0,:]
  print "Time used: ", time.time() - t

def test_dfsti():
  from largescale.src.stimulus import DFStimulus
  import numpy as np
  import cv2
  config = CommonConfig({
    "orientation": np.pi / 6,
    "frequency": 0.01,
    "speed": 100.0,
    "phase": 0.0
  })
  sti = DFStimulus((288,144), config)
  t = 0
  dt = 1
  for i in xrange(10):
    im = sti.get(t)
    im = ((im + 1.0) * 0.5 * 255.0).astype(np.uint8)
    filename = "/home/share/work/outputs/df_%d.tiff" % i
    cv2.imwrite(filename, im)
    t = t + dt

def test_conv2d():
  import largescale.src.support.convolution as conv
  import largescale.src.support.cl_support as clspt
  import time
  import numpy as np
  import cv2
  im = cv2.imread("/home/share/work/outputs/testimg.jpg")
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  imv = clspt.Variable(im.astype(np.float32))
  imo = clspt.Variable(np.empty_like(im).astype(np.float32))
  kernels = [
    np.array([[1,0,-1],[0,0,0],[-1,0,1]]),
    np.array([[-1,0,1],[0,0,0],[1,0,-1]]),
    np.zeros([10,10]).astype(np.float32) + 1.0/100.0
  ]
  nkernels = len(kernels)
  kernels = conv.Conv2DKernelPool(kernels)
  ikernels = np.zeros_like(im).astype(np.int32)
  for i in xrange(im.shape[0]):
    ikernels[i,:] = int(i * nkernels / im.shape[0])
  t = time.time()
  conv.conv2d(imv, imo, kernels, ikernels)
  print "Time used: ", time.time() - t
  imout = imo.fetch()
  imout = imout.astype(np.uint8)
  cv2.imwrite("/home/share/work/outputs/testimg_out.jpg", imout)

def test_map_kernel():
  import largescale.src.support.cl_support as clspt
  import numpy as np
  kern = clspt.map_kernel("a[i] * 1.0 + b[i] * c")
  n = 10
  a = clspt.Variable( np.arange(n).astype(np.float32) + 10.0 )
  b = clspt.Variable( 2.0 - a.buf_host )
  c = 1.0
  r = clspt.Variable( np.zeros_like(a.buf_host) )
  kern(a=a, b=b, c=c, out=r)
  print r.fetch()

def test_model():
  from largescale.src.v1 import V1DirectNeuronGroup, T_E, T_I
  import numpy as np
  config = CommonConfig({
    "tau_rise_gaba1": 0.1,
    "tau_damp_gaba1": 0.1,
    "tau_rise_gaba2": 0.1,
    "tau_damp_gaba2": 0.1,
    "tau_rise_ampa": 0.1,
    "tau_damp_ampa": 0.1,
    "tau_rise_nmda": 0.1,
    "tau_damp_nmda": 0.1
  })

test_geo()
