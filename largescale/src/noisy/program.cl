#include <support/cl_support.cl>

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
  __global double *g_previous, // read buffer
  __global double *g,          // write buffer
  __global double *s_previous, // read buffer
  __global double *s,          // write buffer
  __global double *tspikes,    // spiking times
  double firing_rate,
  double tau_rise,
  double tau_damp,
  double t,
  double dt,
  __global unsigned int *randseeds)
{
  int i = get_global_id(0);
  double g_val = g_previous[i];
  double s_val = s_previous[i];
  unsigned int rndnum = randseeds[i];
  double tspk = tspikes[i];
  double spkitv = tspk - t; // spike interval
  double exp_rise;
  double exp_damp;
  double t_end = t + dt;
  double tau_rise_inv = 1.0 / tau_rise;
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
    s_val = s_val + tau_rise_inv;
    // update the next spike
    rndnum = rand(rndnum);
    spkitv = -log( ((double)rndnum) / ((double)RAND_MAX+1.0) ) / firing_rate;
    tspk += spkitv;
  }
  // update the result
  g[i] = g_val;
  s[i] = s_val;
  tspikes[i] = tspk;
}