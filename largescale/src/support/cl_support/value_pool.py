# Package: largescale.src.support.common.value_pool

import numpy as np
from largescale.src.support.cl_support import Variable

# A value-pool stores a series of values in the
# device buffer. This is for the case that many
# kernels use several different parameters to
# calulate. For a specified value-pool, a kernel
# should also require a value-pool-spec, which
# stores the indexes of the value to use for
# each kernel. For example, if we have two ava-
# ilabel parameters for 100x100 kernels to use,
# then we many have a value-pool like:
#   pool = [ 0.01  0.03 ]
# And if for each 2x2 grid of the kernels, only
# the kernel at the right-bottom corner takes
# the 0.03 parameter (others takes 0.01), then
# the value-pool-spec should look like:
#   spec = [[ 0   0   0   0  ...  0   0 ]
#           [ 0   1   0   1  ...  0   1 ]
#           [ 0   0   0   0  ...  0   0 ]
#           [ 0   1   0   1  ...  0   1 ]
#            ... ... ... ... ... ... ...
#           [ 0   0   0   0  ...  0   0 ]
#           [ 0   1   0   1  ...  0   1 ]]

# Spec of value pool (or other pools)
# Properties:
#   buf:  [Buffer]<int> buffer on device
#   spec: [ndarray]<int> buffer on host
class ValuePoolSpec:
  def __init__(self, spec):
    if isinstance(spec, Variable):
      self.var = spec
    else:
      self.var = Variable( np.array(spec).astype(np.int32), read_only=True )
    self.shape = self.var.shape
  @property
  def spec(self):
    return self.var.buf_host
  @property
  def buf(self):
    return self.var.buf_dev
  @property
  def buf_dev(self):
    return self.var.buf_dev

# Value pool
# Properties:
#   buf:   [Buffer]<float> buffer on device
#   spec:  [ValuePoolSpec] the spec along with this pool
class ValuePool:
  def __init__(self, values, spec = None):
    self.spec = None
    if not spec is None:
      if isinstance(spec, ValuePoolSpec):
        self.spec = spec
      else:
        self.spec = ValuePoolSpec(spec)
    if isinstance(values, Variable):
      self.var = values
    else:
      self.var = Variable( np.array(values), read_only=True )
  
  @property
  def values(self):
    return self.var.buf_host
  @property
  def buf(self):
    return self.var.buf_dev
  @property
  def buf_dev(self):
    return self.var.buf_dev