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