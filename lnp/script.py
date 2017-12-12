# Compare 2 mechanisms of motion illusion
# this script is written in tensorflow

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lnplib import tf_conv1d

def delta_sti(length, tau, c=1000.0):
  sc = 1.0
  div=100.0 # div must be larger than 4
  bw = np.log((div - 2 + np.sqrt(div * (div - 4))) / 2) / c
  if (bw <= 1): sc = 4.0
  else: sc = 1.0
  pk = c / 4.0
  sti = np.array(range(length)).astype(np.float) - tau
  stiexp = tf.exp(-c * sti)
  sti = sc / (1/stiexp + 1) / (1 + stiexp) # don't know why the coefficient is 4, it just works
  return sti

# exponential response curve
def exp_res(length, tau):
  trng = np.array(range(length)).astype(np.float)
  # curve = trng / tau * tf.exp(-trng / tau)
  curve = tf.exp(-trng / tau)
  return curve

# shift the stimuli
def mec1(tau, tauf):
  def fn_on(sti):
    length = int(sti.shape[0])
    rescv = exp_res(length, tauf)
    return tf_conv1d(sti, resc, "SAME")
  def fn_off(sti):
    shape = int(sti.shape[0])
    sti = delta_sti()
    if (tau > 0): sti2[0:tau] = 0
    else: sti2[tau:] = 0
    return sti / tauf * np.exp( - sti / tauf )
  return (fn_on,fn_off)

def mec2(tauf_on, tauf_off):
  def fn_on(sti):
    return sti / tauf * np.exp( - sti / tauf )


sess = tf.Session()

l = 1000
tt = np.array(range(l)).astype(np.float)
sti = delta_sti(l, 10)
resc = exp_res(100, 50)
res = tf_conv1d(sti, resc, "VALID")
resv = sess.run(res)
plt.plot(resv)
plt.show()
