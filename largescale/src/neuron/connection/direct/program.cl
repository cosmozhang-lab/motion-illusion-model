#include <std.cl>
#include <geometry.cl>


/* 
 * Add convolution result input to relaxation item `s`.
 * This function cooperate with `chain2`. In each `dt`,
 * we convolve the stimulus. Then use this function to
 * add the convolution result to the relaxatiion item `s`.
 * And evolve the conductance using `chain2`.
 *
 @param 
 */
__kernel void convinput(
  __global float *amp_pool,
  __global int *amp_specs,
  __global float *conv_sti,
  __global float *s_previous, // read buffer
  __global float *s,          // write buffer
  __global float *tau_rise_pool,
  __global int *tau_rise_indexes
  )
{
  int id = get_global_id(0);
  float amp = amp_pool[amp_specs[id]];
  float conv_sti_val = conv_sti[id];
  float tau_rise = tau_rise_pool[tau_rise_indexes[id]];
  float tau_rise_inv = 1.0 / tau_rise;
  float sval = s_previous[id];
  sval = sval + conv_sti_val * amp * tau_rise_inv;
  s[id] = sval;
}