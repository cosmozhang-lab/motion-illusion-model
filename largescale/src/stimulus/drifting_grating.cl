__kernel void drifting_grating(
  int rows,
  int cols,
  __global double *buf,
  double sin_o,
  double cos_o,
  double frequency,
  double phase,
  double pi
) {
  int id = get_global_id(0);
  int ir = (int)(id / cols);
  int ic = id - ir * cols;
  double x = (double)ic - (double)cols / 2.0;
  double y = (double)ir - (double)rows / 2.0;
  double yy = x * sin_o + y * cos_o;
  double yyy = 2.0 * pi * (yy * frequency) - phase;
  buf[id] = cos(yyy);
}