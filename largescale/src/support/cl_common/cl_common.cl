__kernel void add(__global float *a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] + b[i];
}

__kernel void addscalar(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a + b[i];
}

__kernel void sub(__global float *a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] - b[i];
}

__kernel void subscalar(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a - b[i];
}

__kernel void dotmul(__global float *a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] * b[i];
}

__kernel void timed(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a * b[i];
}

__kernel void inverse(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = 1.0 / b[i];
}

__kernel void minus(__global float *a, __global float *result)
{
  int i = get_global_id(0);
  result[i] = 0.0 - a[i];
}