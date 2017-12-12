import numpy as np
import tensorflow as tf

def generatePoissSequence(firing_rates):
  return [np.random.poisson(fr) for fr in firing_rates]

def tf_conv1d(sequence, filters, padding="SAME"):
  augflt = []
  fltlen = int(filters.shape[0])
  seqlen = int(sequence.shape[0])
  filters = tf.reverse(filters, axis=[0])
  if padding == "SAME":
    outlen = seqlen
    for i in xrange(outlen):
      concats = []
      if i + 1 > fltlen:
        concats.append(np.zeros(i+1-fltlen))
        concats.append(filters)
      else:
        concats.append(filters[-(i+1):])
      if i < seqlen - 1:
        concats.append(np.zeros(seqlen-i-1))
      fltrow = tf.concat(concats, axis=0)
      fltrow = fltrow[:seqlen]
      augflt.append(fltrow)
  elif padding == "VALID":
    outlen = seqlen - fltlen + 1
    for i in xrange(outlen):
      concats = []
      if i + 1 > fltlen:
        concats.append(np.zeros(i+1-fltlen))
        concats.append(filters)
      else:
        concats.append(filters[-(i+1):])
      if i < seqlen - 1:
        concats.append(np.zeros(seqlen-i-1))
      fltrow = tf.concat(concats, axis=0)
      fltrow = fltrow[:seqlen]
      augflt.append(fltrow)
  elif padding == "FULL":
    outlen = seqlen + fltlen - 1
    for i in xrange(outlen):
      concats = []
      concats.append(np.zeros(i))
      concats.append(filters)
      if i + fltlen < seqlen:
        concats.append(np.zeros(seqlen-i-fltlen))
      fltrow = tf.concat(concats, axis=0)
      fltrow = fltrow[:seqlen]
      augflt.append(fltrow)
  else:
    return None
  augflt = tf.stack(augflt, axis=1)
  return tf.reshape(tf.matmul(tf.reshape(sequence,[1,seqlen]), augflt), [outlen])
