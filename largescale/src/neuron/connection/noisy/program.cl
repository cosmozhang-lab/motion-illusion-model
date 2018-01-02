#include <std.cl>

/* 
 * Calcuate the conductance decaying process
 * for noisy input group. Each neuron may
 * spike several times in `dt`. We simulate
 * this process by taking following process:
 *     `chain2` on `t` to `t_spike_1`
 *     `s` = `s` + `amp` / `tau_r`
 *     `chain2` on `t_spike_1` to `t_spike_2`
 *     `s` = `s` + `amp` / `tau_r`
 *     ...
 *     `chain2` on `t_spike_last` to `t + dt`
 * The `chain2` process can be expressed as ODEs:
 *     tau_damp * dg/dt = - g + s
 *     tau_rise * ds/dt = - s
 * The spikes stream obeys a poisson progress: 
 *     P[N(t+tau)-N(t)=k] = 
 *         exp(-lambda*tau) * (lambda*tau)^k / k!
 * So the spike interval `tau` obeys distribution:
 *     pdf(tau) = P[N(t+tau)-N(t)=k]
 *              = (lambda*tau) * exp(-lambda*tau)
 * @param g:           the conductances
 * @param s:           the relaxation items (ds/dt receives the spike pulse directly)
 * @param tspikes:     spiking times (this buffer is used for both read and write)
 * @param firing_rate: the noisy firing rate
 * @param tau_rise:    time constance of conductance rising
 * @param tau_damp:    time constance of conductance damping
 * @param t:           start time
 * @param dt:          delta time
 * @param randseeds:   random seed
 */
__kernel void chain2noisy(
  __global float *g_previous, // read buffer
  __global float *g,          // write buffer
  __global float *s_previous, // read buffer
  __global float *s,          // write buffer
  __global float *tspikes,    // spiking times
  __global float *amp_pool,
  __global int *amp_specs,
  __global float *firing_rate_pool,
  __global int *firing_rate_specs,
  __global float *tau_rise_pool,
  __global int *tau_rise_specs,
  __global float *tau_damp_pool,
  __global int *tau_damp_specs,
  float t,
  float dt,
  __global unsigned int *randseeds)
{
  int i = get_global_id(0);
  float g_val = g_previous[i];
  float s_val = s_previous[i];
  float firing_rate = firing_rate_pool[firing_rate_specs[i]];
  float tau_rise = tau_rise_pool[tau_rise_specs[i]];
  float tau_damp = tau_damp_pool[tau_damp_specs[i]];
  float amp = amp_pool[amp_specs[i]];
  unsigned int rndnum = randseeds[i];
  float tspk = tspikes[i];
  float spkitv = tspk - t; // spike interval
  float exp_rise;
  float exp_damp;
  float t_end = t + dt;
  float tau_rise_inv = 1.0 / tau_rise;
  while (true) {
    // If next spike is beyond this time bin, we only
    // process to the end of this time bin.
    if (tspk > t_end) spkitv -= tspk - t_end;
    // process to next spike
    if (spkitv > 0) {
      exp_rise = exp(- spkitv / tau_rise);
      exp_damp = exp(- spkitv / tau_damp);
      g_val = g_val * exp_damp + tau_rise * (exp_damp - exp_rise) / (tau_damp - tau_rise) * s_val;
      s_val = s_val * exp_rise;
    }
    // If next spike is beyond this time bin, we stop
    if (tspk > t_end) break;
    // add next spike input
    s_val = s_val + amp * tau_rise_inv;
    // update the next spike
    rndnum = rand(rndnum);
    spkitv = -logf( ((float)rndnum) / ((float)RAND_MAX+1.0) ) / firing_rate;
    tspk += spkitv;
  }
  // update the result
  g[i] = g_val;
  s[i] = s_val;
  tspikes[i] = tspk;
}