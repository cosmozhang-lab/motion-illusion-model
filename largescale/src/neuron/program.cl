/* 
 * Calcuate the conductance decaying process
 * The process can be expressed as ODEs:
 *     tau_damp * dg/dt = - g + s
 *     tau_rise * ds/dt = - s + sum( delta(t - t_spike) )
 * @param g:        the conductances
 * @param s:        the relaxation items (ds/dt receives the spike pulse directly)
 * @param tau_rise: time constance of conductance rising
 * @param tau_damp: time constance of conductance damping
 * @param dt:       delta time
 */
__kernel void chain2(__global double *g, __global double *s, double tau_rise, double tau_damp, double dt)
{
  double tr = dt / tau_rise;
  double etr = exp(-tr);
  double td = dt / tau_damp;
  double etd = exp(-td);
  double cst = tau_rise / (tau_damp - tau_rise) * (etd - etr);
  int i = get_global_id(0);
  g[i] = g[i] * etd + cst * s[i];
  s[i] = s[i] * etr;
  return;
}