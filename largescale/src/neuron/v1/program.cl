__kernel void rk2params(
  __global const float *g_lgn_on,
  __global const float *g_lgn_off,
  __global const float *g_noise_gaba1,
  __global const float *g_noise_gaba2,
  __global const float *g_noise_nmda,
  __global const float *g_noise_ampa,
  float f_gaba_noise,
  float f_nmda_noise,
  float v_exc,
  float v_inh,
  __global float *alpha,
  __global float *beta
) {
  int i = get_global_id(0);
  float g_lgn = g_lgn_on[i] + g_lgn_off[i];
  float g_noise_inh = (1 - f_gaba_noise) * g_noise_gaba2[i] + f_gaba_noise * g_noise_gaba1[i];
  float g_noise_exc = (1 - f_nmda_noise) * g_noise_ampa[i] + f_nmda_noise * g_noise_nmda[i];
  //alpha0[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
  //beta0[i] = (gl[i] + gexcit + genoise) * vexcit + (ginoise + ginhib) * vinhib;
  alpha[i] = - g_lgn - g_noise_inh - g_noise_exc;
  beta[i] = (g_lgn + g_noise_exc) * v_exc + (g_noise_inh) * v_inh;
}