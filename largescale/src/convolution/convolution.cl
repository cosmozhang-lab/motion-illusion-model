#include <support/geometry.cl>

/**
 * 2D Convolution
 * Using "same" padding principle
 * For detailed comment, refer to convolution.py
 */
__kernel void conv2d(
  int rows,
  int cols,
  __global float *input_map,
  __global float *kernels,
  __global int *kernel_shapes,
  __global int *ikernels,
  __global float *output_map
) {
  int id = get_global_id(0);
  int ir = (int)(id / cols);
  int ic = id - ir * cols;
  int ik = ikernels[id];
  long long kern_offset = 0;
  for (int i = 0; i < ik; i++) kern_offset += kernel_shapes[i*2] * kernel_shapes[i*2+1];
  // kernel size
  int krows = kernel_shapes[ik*2];
  int kcols = kernel_shapes[ik*2+1];
  // half of kernel size
  int hkrows = krows / 2;
  int hkcols = kcols / 2;
  // kernel left-top coordinate in input_map
  int ktop  = ir + hkrows - krows + 1;
  int kleft = ic + hkcols - kcols + 1;
  // start to convolve
  float res = 0.0;
  for (int r = 0; r < krows; r++) {
    int mr = ktop + r;
    if (mr < 0) continue;
    if (mr >= rows) break;
    for (int c = 0; c < kcols; c++) {
      int mc = kleft + c;
      if (mc < 0) continue;
      if (mc >= cols) break;
      int ipx = coor2idx2d(rows, cols, mr, mc);
      int ipxk = coor2idx2d(krows, kcols, r, c);
      res += kernels[kern_offset + ipxk] * input_map[ipx];
    }
  }
  output_map[ coor2idx2d(rows, cols, ir, ic) ] = res;
}
