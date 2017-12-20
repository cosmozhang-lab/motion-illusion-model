/* 
 * Calcuate the conductance decaying process.
 * This process do not involve spike inputs,
 * which is different from `chain2_with_input`
 * in `connection/program.cl`.
 * The process can be expressed as ODEs:
 *     tau_damp * dg/dt = - g + s
 *     tau_rise * ds/dt = - s
 * @param g:        the conductances
 * @param s:        the relaxation items (ds/dt receives the spike pulse directly)
 * @param tau_rise: time constance of conductance rising
 * @param tau_damp: time constance of conductance damping
 * @param dt:       delta time
 * @param exp_rise: exp( -dt / tau_rise ). This should be computed on CPU for better precision
 * @param exp_damp: exp( -dt / tau_damp ). This should be computed on CPU for better precision
 */
__kernel void chain2(
  __global double *g_previous, // read buffer
  __global double *g,          // write buffer
  __global double *s_previous, // read buffer
  __global double *s,          // write buffer
  double tau_rise,
  double tau_damp,
  double dt,
  double exp_rise,
  double exp_damp)
{
  int i = get_global_id(0);
  double g_val = g_previous[i];
  double s_val = s_previous[i];
  //double tr = dt / tau_rise;
  //double etr = exp_rise;
  //double td = dt / tau_damp;
  //double etd = exp_damp;
  //double cst = tau_rise / (tau_damp - tau_rise) * (etd - etr);
  //g[i] = g_val * etd + cst * s_val;
  //s[i] = s_val * etr;
  g[i] = g_val * exp_damp + tau_rise * (exp_damp - exp_rise) / (tau_damp - tau_rise) * s_val;
  s[i] = s_val * exp_rise;
}