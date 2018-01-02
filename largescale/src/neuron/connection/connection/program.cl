#include <std.cl>
#include <geometry.cl>


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
  __global int *cnctmap,
  __global float *s_previous, // read buffer
  __global float *s,          // write buffer
  __global float *tau_rise_pool,
  __global int *tau_rise_indexes
  )
{
  int ispkr = (int)(ispike / cols); // row index of the spiking neuron
  int ispkc = ispike - ispkr * cols; // column index of the spiking neuron
  int id = get_global_id(0);
  int ir = (int)(id / cols); // row index of neuron
  int ic = id - ir * cols; // column index of neuron
  if (cnctmap && (cnctmap[id] != cnctmap[ispike])) return;
  float tau_rise = tau_rise_pool[tau_rise_indexes[id]];
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
  int top = MAX(MAX(ktop, ntop), 0);
  int bottom = MIN(MIN(kbottom, nbottom), rows);
  int left = MAX(MAX(kleft, nleft), 0);
  int right = MIN(MIN(kright, nright), cols);
  // do iteration
  if (ispkc >= left && ispkc < right && ispkr >= top && ispkr < bottom) {
    float kern_val = kern[ coor2idx2d(krows, kcols, ispkr - ktop, ispkc - kleft) ];
    float cnct_val = cnct_offset[ coor2idx2d(nrows, ncols, ispkr - ntop, ispkc - nleft) ];
    float amp = kern_val * cnct_val;
    float s_val = s[id];
    float tau_rise_inv = 1.0 / tau_rise;
    s[id] = s_val + tau_rise_inv * amp;
  }
}