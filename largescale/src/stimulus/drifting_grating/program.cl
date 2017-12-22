__kernel void drifting_grating(
  int rows,
  int cols,
  __global float *buf,
  float sin_o,
  float cos_o,
  float frequency,
  float phase,
  float pi
) {
  int id = get_global_id(0);
  int ir = (int)(id / cols);
  int ic = id - ir * cols;
  float x = (float)ic - (float)cols / 2.0;
  float y = (float)ir - (float)rows / 2.0;
  float yy = x * sin_o + y * cos_o;
  float yyy = 2.0 * pi * (yy * frequency) - phase;
  buf[id] = cos(yyy);
}