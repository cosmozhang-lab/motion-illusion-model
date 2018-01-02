# Package: largescale.src.support.plots.colormap

import matplotlib.colors

def hsv2rgb(hsv):
  (h,s,v) = hsv
  h = float(h)
  s = float(s)
  v = float(v)
  hi = int(h/60.0) % 6
  f = h/60.0 - hi
  p = v * (1 - s)
  q = v * (1 - f * s)
  t = v * (1 - (1 - f) * s)
  rgb = {
    0: (v,t,p),
    1: (q,v,p),
    2: (p,v,t),
    3: (p,q,v),
    4: (t,p,v),
    5: (v,p,q)
  }[hi]
  return tuple([max(min(c,1.0),0.0) for c in rgb])

def color_float2uint(val):
  return tuple([max(min(int(c*256.0),255),0) for c in val])

def color_uint2str(val):
  return "#%02x%02x%02x" % val

def circle_colormap(subdivisions=256):
  cdict = [hsv2rgb((float(i) / subdivisions * 360.0, 1, 1)) for i in xrange(subdivisions)]
  return matplotlib.colors.ListedColormap(cdict, 'indexed')