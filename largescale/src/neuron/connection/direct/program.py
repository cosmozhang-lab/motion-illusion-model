import numpy as np
import largescale.src.support.cl_support as clspt
from largescale.src.support.common import CommonConfig
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "program.cl") )

kernel_convinput = program.convinput.kernel
def convinput(s, amp_pool, conv_sti, tau_rise_pool, update=True, queue=None):
  queue = queue or clspt.queue()
  kernel_convinput(queue, (s.size,), None, amp_pool.buf, amp_pool.spec.buf, conv_sti.buf_dev, s.buf_dev, s.buf_swp, tau_rise_pool.buf, tau_rise_pool.spec.buf)
  if update:
    s.update(queue)