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
  __global float *g_previous, // read buffer
  __global float *g,          // write buffer
  __global float *s_previous, // read buffer
  __global float *s,          // write buffer
  __global float *tau_rise_pool,
  __global int *tau_rise_specs,
  __global float *tau_damp_pool,
  __global int *tau_damp_specs,
  float dt,
  float exp_rise,
  float exp_damp)
{
  int i = get_global_id(0);
  float g_val = g_previous[i];
  float s_val = s_previous[i];
  float tau_rise = tau_rise_pool[tau_rise_specs[i]];
  float tau_damp = tau_damp_pool[tau_damp_specs[i]];
  //float tr = dt / tau_rise;
  //float etr = exp_rise;
  //float td = dt / tau_damp;
  //float etd = exp_damp;
  //float cst = tau_rise / (tau_damp - tau_rise) * (etd - etr);
  //g[i] = g_val * etd + cst * s_val;
  //s[i] = s_val * etr;
  g[i] = g_val * exp_damp + tau_rise * (exp_damp - exp_rise) / (tau_damp - tau_rise) * s_val;
  s[i] = s_val * exp_rise;
}






/**
 * Use RK2 algorithm to calculate the voltage
 * evolution and the spikes.
 * Voltage evolves according to ODE:
 *   dv/dt = sum<i>{ - g_i * (v - v_ref_i) }
 *   Where `i` represents either `leak`, `in-
 *   hibitory` or `excitatory`. `v_ref_i` re-
 *   presents the refactory votage.
 *   and:
 *     g_inh = g_inh_recurrent + g_inh_noisy
 *     g_exc = g_exc_recurrent + g_exc_noisy
 *             + g_input
 * As we use RK2 algorithm, `v(t+dt)` can be
 * calculated as:
 *   v(t+dt) = v(t) + k * dt
 *   where:
 *   k = ( d{v(t)}/dt + {d{v(t)+k1*dt}/dt}_t ) / 2
 *   where:
 *   k1 = {dv/dt}_t
 *      = sum<i>{ - g_i(t) * (v(t) - v_ref_i) }
 *   k2 = {dv/dt}_t
 *      = sum<i>{ - g_i(t+dt) * (v(t) + k1*dt - v_ref_i) }
 * Note that if neuron spiked at `ts`, then 
 * voltage will be reset to a refactory vo-
 * ltage `v_reset`, and hold this level for
 * at least a refactory time `t_ref`. And
 * also note that neuron spikes immediately
 * when `v` reaches threshold `v_thre`. So
 * we must:
 * 1) Evolve `v` from time point `ts+t_ref`
 *    if it is between `t` and `t+dt`, ins-
 *    tead of evolving from `t`.
 * 2) reset `v` to `v_reset` at `ts` if `ts`
 *    is between `t` and `t+dt`. In which:
 *      ts = t + (v_thre - v) / k
 *    And then hold `v_reset` until `ts +
 *    t_ref` or to the end of the time bin
 *    `t + dt`.
 * As we will only have `g_i` at the start
 * and the end of the time bin, according
 * to the conductance evolving stage, we
 * cannot make it so precise that the evo-
 * lving start time is calculated conside-
 * ring all the above conditions. So for
 * the `g_i`s in the equations we can only
 * take them as:
 *   g_i(t) = gi0
 *   g_i(t+dt) = gi1
 * where `gi0` represents `g_i` at the
 * start of the time bin, and `gi1` that
 * at the end of the time bin.
 * Besides, to reduce the number of params
 * of this RK2 function, we transform the
 * equations as:
 *   k1 = - sum<i>{ gi0 } * v(t) + sum<i>{ gi0 * v_ref_i }
 *   k1 = - sum<i>{ gi1 } * (v(t)+k1*dt) + sum<i>{ gi1 * v_ref_i }
 * so we can represent all the conductance
 * params (i.e. `gi`s) into four params:
 *   alpha0 = - sum<i>{ gi0 }
 *   beta0 = sum<i>{ gi0 * v_ref_i }
 *   alpha1 = - sum<i>{ gi1 }
 *   beta1 = sum<i>{ gi1 * v_ref_i }
 * and the above equations transformed to:
 *   k1 = alpha0 * v(t) + beta0
 *   k2 = alpha1 * (v(t)+k1*dt) + beta1
 * NOTE: `t_ref` must be larger than `dt`
 * so that neuron spikes only once in a
 * time bin.
 * @param v:                   voltage of each neuron
 * @param tspikes:             last spike times
 * @param t_refs:              refactory time for each neuron
 * @param v_thre:              voltage spike threshold
 * @param v_reset:             voltage in refactory period
 * @param alpha0:              alpha params at `t`
 * @param beta0:               beta params at `t`
 * @param alpha1:              alpha params at `t` + `dt`
 * @param beta1:               beta params at `t` + `dt`
 * @param t:                   time bin start
 * @param dt:                  delta time
 */
__kernel void rk2voltage(
  __global float *v_previous, // read buffer
  __global float *v,          // write buffer
  __global float *tspikes,
  __global float *t_ref_pool,
  __global int *t_ref_specs,
  __global float *alpha0,
  __global float *beta0,
  __global float *alpha1,
  __global float *beta1,
  float v_thre,
  float v_reset,
  float t,
  float dt
) {
  int i = get_global_id(0);
  float v_val = v_previous[i];
  float tref = t_ref_pool[t_ref_specs[i]];
  float tspk = tspikes[i];
  float holdtime = tspk + tref - t;
  float t_start = t;
  float t_end = t + dt;
  float a0 = alpha0[i];
  float b0 = beta0[i];
  float a1 = alpha1[i];
  float b1 = beta1[i];
  if (holdtime > 0) {
    // if we are still in refactory period
    // at the start of the time bin
    t_start = t + holdtime;
    v_val = v_reset;
  }
  if (t_start < t_end) {
    // if in this time bin we have exited
    // the refactory period, we start ev-
    // olution.
    float ddt = t_end - t_start;
    float k1 = a0 * v_val + b0;
    float k2 = a1 * (v_val + k1 * ddt) + b1;
    float k = (k1 + k2) * 0.5;
    float v_new = v_val + k * ddt;
    if (v_new > v_thre) {
      // Voltage has rised beyond the threshold.
      // Neuron spikes in this time bin. And will
      // hold at `v_reset` to the end of the time
      // bin.
      ddt = (v_thre - v_val) / k;
      tspk = t_start + ddt;
      v_new = v_reset;
      tspikes[i] = tspk;
    }
  }
}