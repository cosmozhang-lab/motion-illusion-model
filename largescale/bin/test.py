# Unit testings

def test_v1():
  from largescale.src.v1 import V1DirectNeuronGroup
  import numpy as np
  config = {
    "tau_rise_gaba1": 0.1,
    "tau_damp_gaba1": 0.1,
    "tau_rise_gaba2": 0.1,
    "tau_damp_gaba2": 0.1,
    "tau_rise_ampa": 0.1,
    "tau_damp_ampa": 0.1,
    "tau_rise_nmda": 0.1,
    "tau_damp_nmda": 0.1
  }
  n = V1DirectNeuronGroup((288,144), config = config)
  import time
  t = time.time()
  dt = 0.001
  ts = np.arange(1000).astype(np.double) * dt
  for tt in ts:
    n.step(tt, dt)
  print time.time() - t


def test_cl():
  import pyopencl as cl
  import largescale.src.cl_support as clspt
  import numpy as np
  import time
  n = 100000
  ctx = clspt.context()
  prg = cl.Program(ctx, """
  __kernel void sum(__global const double *a_g, __global const double *b_g, __global double *res_g)
  {
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] + b_g[gid];
    for (int i = 0; i < 1000000; i++)
      res_g[gid] += a_g[gid] + b_g[gid];
  }
  """).build()
  mf = cl.mem_flags
  queues = []
  a_gs = []
  b_gs = []
  res_gs = []
  ni = 2
  for i in xrange(ni):
    a_np = np.random.rand(n).astype(np.double)
    b_np = np.random.rand(n).astype(np.double)
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
  # cl.enqueue_copy(queues[0], res_np, res_gs[0])
  cl.enqueue_barrier(queues[-1])
  print time.time() - t

test_v1()
