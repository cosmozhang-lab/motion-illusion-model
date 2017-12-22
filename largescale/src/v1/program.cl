__kernel void rk2params(
  __global const double *g_lgn_on,
  __global const double *g_lgn_off,
  __global const double *g_noise_inh_gaba1,
  __global const double *g_noise_inh_gaba2,
  __global const double *g_noise_exc_nmda,
  __global const double *g_noise_exc_ampa,
  double f_gaba_noise,
  double f_nmda_noise,
  double v_exc,
  double v_inh,
  __global double *alpha,
  __global double *beta
) {
  int i = get_global_id(0);
  double g_lgn = g_lgn_on[i] + g_lgn_off[i];
  double g_noise_inh = (1 - f_gaba_noise) * g_noise_inh_gaba2[i] + f_gaba_noise * g_noise_inh_gaba1[i];
  double g_noise_exc = (1 - f_nmda_noise) * g_noise_exc_ampa[i] + f_nmda_noise * g_noise_exc_nmda[i];
  //alpha0[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
  //beta0[i] = (gl[i] + gexcit + genoise) * vexcit + (ginoise + ginhib) * vinhib;
  alpha[i] = - g_lgn - g_noise_inh - g_noise_exc;
  beta[i] = (g_lgn + g_noise_exc) * v_exc + (g_noise_inh) * v_inh;
}