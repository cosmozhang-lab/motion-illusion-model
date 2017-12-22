#include <support/cl_support.cl>
#include <support/geometry.cl>

/* 
 * Calcuate the conductance decaying process with spike input.
 * This process do involve spike inputs,
 * which is different from `chain2` in `neuron/program.cl`.
 * The process can be expressed as ODEs:
 *     tau_damp * dg/dt = - g + s
 *     tau_rise * ds/dt = - s + sum( delta(t - t_spike) )
 *
 * @Remark: I don't think this is necessary. As we can use
 * chain2 without input, just add `amp` / tau_r to `s` on each
 * spike. This does make sense. The process is:
 *     `chain2` on `t` to `t_spike_1`
 *     `s` = `s` + `amp` / `tau_r`
 *     `chain2` on `t_spike_1` to `t_spike_2`
 *     `s` = `s` + `amp` / `tau_r`
 *     ...
 *     `chain2` on `t_spike_last` to `t + dt`
 *
 * @param rows:        size of neuron map - rows
 * @param cols:        size of neuron map - columns
 * @param ispike:      index of the spiking neuron
 * @param krows:       size of neuron map - rows
 * @param kcols:       size of neuron map - columns
 * @param kern:        the connection kernel
 * @param icncts:      index of connectivity to use in the connectivity pool
 * @param cnct_shapes: shapes of the connectivity matrixes: [rows_1, cols_1, rows_2, cols_2, ...]
 * @param cncts:       connectivity pool data
 * @param g:           the conductances
 * @param s:           the relaxation items (ds/dt receives the spike pulse directly)
 * @param tau_rise:    time constance of conductance rising
 * @param tau_damp:    time constance of conductance damping
 * @param dt:          delta time
 * @param exp_rise:    exp( -dt / tau_rise ). This should be computed on CPU for better precision
 * @param exp_damp:    exp( -dt / tau_damp ). This should be computed on CPU for better precision
 */
__kernel void chain2_with_input(
  int rows,
  int cols,
  int ispike,
  int krows,
  int kcols,
  __global float *kern,
  __global int *icncts,
  __global int *cnct_shapes,
  __global float *cncts,
  __global float *g_previous, // read buffer
  __global float *g,          // write buffer
  __global float *s_previous, // read buffer
  __global float *s,          // write buffer
  float tau_rise,
  float tau_damp,
  float dt,
  float exp_rise,
  float exp_damp
  )
{
  int ispkr = (int)(ispike / cols); // row index of the spiking neuron
  int ispkc = ispike - ispkr * cols; // column index of the spiking neuron
  int id = get_global_id(0);
  int ir = (int)(id / cols); // row index of neuron
  int ic = id - ir * cols; // column index of neuron
  int in = icncts[id]; // index of connectivity matrix
  __global float * cnct_offset = cncts;
  for (int i = 0; i < in; i++) cnct_offset += cnct_shapes[i*2] * cnct_shapes[i*2+1];
  // connectivity matrix size
  int nrows = cnct_shapes[in*2];
  int ncols = cnct_shapes[in*2+1];
  // half of connectivity matrix size
  int hnrows = nrows / 2;
  int hncols = ncols / 2;
  // connectivity matrix left-top coordinate in input_map
  int ntop  = ir + hnrows - nrows + 1;
  int nleft = ic + hncols - ncols + 1;
  int nbottom = ntop + nrows;
  int nright = nleft + ncols;
  // half of kernel size
  int hkrows = krows / 2;
  int hkcols = kcols / 2;
  // kernel left-top coordinate in input_map
  int ktop  = ir + hkrows - krows + 1;
  int kleft = ic + hkcols - kcols + 1;
  int kbottom = ktop + krows;
  int kright = kleft + kcols;
  // determine the actual convolving area
  int top = MAX(MAX(ktop, ntop), 0)
  int bottom = MIN(MIN(kbottom, nbottom), rows)
  int left = MAX(MAX(kleft, nleft), 0)
  int right = MIN(MIN(kright, nright), cols)
  // do iteration
  if (ispkc >= left && ispkc < right && ispkr >= top && ispkr < bottom) {
    float kern_val = kern[ coor2idx2d(krows, kcols, ispkr - ktop, ispkc - kleft) ];
    float cnct_val = cnct_offset[ coor2idx2d(nrows, ncols, ispkr - ntop, ispkc - nleft) ];
    float amp = kern_val * cnct_val;
    float g_val = g_previous[id];
    float s_val = s_previous[id];
    tau_rise_inv = 1.0 / tau_rise;
    tau_damp_inv = 1.0 / tau_damp;
    g[id] = exp_damp * g_val + (exp_damp - exp_rise) / (tau_rise_inv - tau_damp_inv) * (tau_rise_inv * tau_damp_inv * amp + tau_damp_inv) * s_val;
    s[id] = exp_rise * s_val + exp_rise * tau_rise_inv * amp;
  }
}

/* 
 * Add spike input to relaxation item `s`. This function 
 * cooperate with `chain2`. In each `dt`, if there are
 * spikes in `dt, we should do `chain2` on each interval
 * between each two contiguous spikes. And between each
 * two contiguous `chain2` processes, we add the spike
 * input to relaxation item with this function(`input`).
 * The process is described as:
 *     `chain2` on `t` to `t_spike_1`
 *     `input` for `spike_1` ( `s` = `s` + `amp` / `tau_r` )
 *     `chain2` on `t_spike_1` to `t_spike_2`
 *     `input` for `spike_2` ( `s` = `s` + `amp` / `tau_r` )
 *     ...
 *     `chain2` on `t_spike_last` to `t + dt`
 *
 * @param rows:        size of neuron map - rows
 * @param cols:        size of neuron map - columns
 * @param ispike:      index of the spiking neuron
 * @param krows:       size of neuron map - rows
 * @param kcols:       size of neuron map - columns
 * @param kern:        the connection kernel
 * @param icncts:      index of connectivity to use in the connectivity pool
 * @param cnct_shapes: shapes of the connectivity matrixes: [rows_1, cols_1, rows_2, cols_2, ...]
 * @param cncts:       connectivity pool data
 * @param s:           the relaxation items (ds/dt receives the spike pulse directly)
 * @param tau_rise:    time constance of conductance rising
 */
__kernel void input(
  int rows,
  int cols,
  int ispike,
  int krows,
  int kcols,
  __global float *kern,
  __global int *icncts,
  __global int *cnct_shapes,
  __global float *cncts,
  __global float *s_previous, // read buffer
  __global float *s,          // write buffer
  float tau_rise
  )
{
  int ispkr = (int)(ispike / cols); // row index of the spiking neuron
  int ispkc = ispike - ispkr * cols; // column index of the spiking neuron
  int id = get_global_id(0);
  int ir = (int)(id / cols); // row index of neuron
  int ic = id - ir * cols; // column index of neuron
  int in = icncts[id]; // index of connectivity matrix
  __global float * cnct_offset = cncts;
  for (int i = 0; i < in; i++) cnct_offset += cnct_shapes[i*2] * cnct_shapes[i*2+1];
  // connectivity matrix size
  int nrows = cnct_shapes[in*2];
  int ncols = cnct_shapes[in*2+1];
  // half of connectivity matrix size
  int hnrows = nrows / 2;
  int hncols = ncols / 2;
  // connectivity matrix left-top coordinate in input_map
  int ntop  = ir + hnrows - nrows + 1;
  int nleft = ic + hncols - ncols + 1;
  int nbottom = ntop + nrows;
  int nright = nleft + ncols;
  // half of kernel size
  int hkrows = krows / 2;
  int hkcols = kcols / 2;
  // kernel left-top coordinate in input_map
  int ktop  = ir + hkrows - krows + 1;
  int kleft = ic + hkcols - kcols + 1;
  int kbottom = ktop + krows;
  int kright = kleft + kcols;
  // determine the actual convolving area
  int top = MAX(MAX(ktop, ntop), 0)
  int bottom = MIN(MIN(kbottom, nbottom), rows)
  int left = MAX(MAX(kleft, nleft), 0)
  int right = MIN(MIN(kright, nright), cols)
  // do iteration
  if (ispkc >= left && ispkc < right && ispkr >= top && ispkr < bottom) {
    float kern_val = kern[ coor2idx2d(krows, kcols, ispkr - ktop, ispkc - kleft) ];
    float cnct_val = cnct_offset[ coor2idx2d(nrows, ncols, ispkr - ntop, ispkc - nleft) ];
    float amp = kern_val * cnct_val;
    float s_val = s[id];
    tau_rise_inv = 1.0 / tau_rise;
    s[id] = s_val + tau_rise_inv * amp;
  }
}