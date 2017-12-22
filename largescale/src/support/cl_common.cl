__kernel add(__global float *a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] + b[i];
}

__kernel addscalar(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a + b[i];
}

__kernel sub(__global float *a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] - b[i];
}

__kernel subscalar(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a - b[i];
}

__kernel dotmul(__global float *a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] * b[i];
}

__kernel timed(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a * b[i];
}

__kernel inverse(float a, __global float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = 1.0 / b[i];
}

__kernel minus(__global float *a, __global float *result)
{
  int i = get_global_id(0);
  result[i] = 0.0 - a[i];
}