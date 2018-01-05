import numpy as np
import pyopencl as cl
import largescale.src.support.cl_support as clspt
import os

thisdir = os.path.split(os.path.realpath(__file__))[0]
program = clspt.compile( os.path.join(thisdir, "program.cl") )

kernel_rk2params = program.rk2params.kernel
def rk2params(g_lgn_on_pos, g_lgn_on_neg, g_lgn_off_pos, g_lgn_off_neg, g_noisy_gaba1, g_noisy_gaba2, g_noisy_nmda, g_noisy_ampa, fgaba_noise, fnmda_noise, v_exc, v_inh, alpha, beta, update=True, queue=clspt.queue()):
  kernel_rk2params(queue, (g_lgn_off_pos.size,), None, g_lgn_on_pos.buf_dev, g_lgn_on_neg.buf_dev, g_lgn_off_pos.buf_dev, g_lgn_off_neg.buf_dev, g_noisy_gaba1.buf_dev, g_noisy_gaba2.buf_dev, g_noisy_nmda.buf_dev, g_noisy_ampa.buf_dev, fgaba_noise, fnmda_noise, v_exc, v_inh, alpha.buf_swp, beta.buf_swp)
  if update:
    alpha.update(queue)
    beta.update(queue)
