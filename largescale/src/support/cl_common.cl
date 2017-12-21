__kernel add(__global double *a, __global double *b, __global double *result)
{
  int i = get_global_id(0);
  result[i] = a[i] + b[i];
}

__kernel sub(__global double *a, __global double *b, __global double *result)
{
  int i = get_global_id(0);
  result[i] = a[i] - b[i];
}

__kernel dotmul(__global double *a, __global double *b, __global double *result)
{
  int i = get_global_id(0);
  result[i] = a[i] * b[i];
}

__kernel timed(double a, __global double *b, __global double *result)
{
  int i = get_global_id(0);
  result[i] = a * b[i];
}

__kernel inverse(double a, __global double *b, __global double *result)
{
  int i = get_global_id(0);
  result[i] = 1.0 / b[i];
}

__kernel minus(__global double *a, __global double *result)
{
  int i = get_global_id(0);
  result[i] = 0.0 - a[i];
}