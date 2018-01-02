import numpy as np
from largescale.src.support.common import CommonConfig
from largescale.src.neuron import V1DirectNeuronGroup, T_EXC, T_INH
from largescale.src.support.geometry import gen_coordinates
from largescale.src.support.geometry.gabor import make_gabor

class PinwheelNetwork:
  def __init__(self, config = CommonConfig()):
    hypercolumn_size = config.fetch("hypercolumn_size", 48)
    if not isinstance(hypercolumn_size, tuple): hypercolumn_size = (hypercolumn_size, hypercolumn_size)
    hypercolumn_narrow_size = min(hypercolumn_size)
    grid_size = config.fetch("grid_size", (6,3))
    cluster_num = config.fetch("cluster_num", 12)
    exc_ratio = config.fetch("exc_ratio", 0.75)
    connectivity_ratio = config.fetch("connectivity_ratio", 0.1)
    connectivity_num = config.fetch("connectivity_num", 6)
    connectivity_size = config.fetch("connectivity_size", (hypercolumn_size[0]*2, hypercolumn_size[1]*2))
    connectivity_scale = config.fetch("connectivity_scale", hypercolumn_narrow_size)
    connectivity_peak = config.fetch("connectivity_peak", 1)
    connectivity_ratio_lr = config.fetch("connectivity_ratio_lr", 0.1)
    connectivity_num_lr = config.fetch("connectivity_num_lr", 6)
    connectivity_size_lr = config.fetch("connectivity_size_lr", (hypercolumn_size[0]*2, hypercolumn_size[1]*2))
    connectivity_scale_lr = config.fetch("connectivity_scale_lr", hypercolumn_narrow_size)
    connectivity_peak_lr = config.fetch("connectivity_peak_lr", 1)
    gabor_size = config.fetch("gabor_size", (hypercolumn_size[0]*2, hypercolumn_size[1]*2))
    gabor_scale = config.fetch("gabor_scale", hypercolumn_narrow_size * 0.2)
    gabor_period = config.fetch("gabor_period", hypercolumn_narrow_size * 0.2)
    gabor_peak = config.fetch("gabor_peak", 1.0)
    delta_orientation = 360.0 / cluster_num
    orientations = [i*delta_orientation for i in xrange(cluster_num)]
    gabor_kernels = [make_gabor(size=gabor_size, orientation=o, scale=gabor_scale, period=gabor_period, peak=gabor_peak) for o in orientations]
    (coory, coorx) = gen_coordinates(hypercolumn_size, center_zero=True)
    prefer_cluster_lt = np.arctan(coory/coorx).astype(np.float32) / np.pi * 180.0 + 90.0
    prefer_cluster_lt[coorx<0] += 180.0
    prefer_cluster_rt = np.fliplr(prefer_cluster_lt)
    prefer_cluster_lb = np.flipud(prefer_cluster_lt)
    prefer_cluster_rb = np.flipud(prefer_cluster_rt)
    prefer_cluster_singles = [[prefer_cluster_lt,prefer_cluster_rt],[prefer_cluster_lb,prefer_cluster_rb]]
    self.shape = (hypercolumn_size[0]*grid_size[0], hypercolumn_size[1]*grid_size[1])
    prefer_clusters = np.zeros(self.shape).astype(np.float32)
    for i in xrange(grid_size[0]):
      for j in xrange(grid_size[1]):
        prefer_clusters[(i*hypercolumn_size[0]):((i+1)*hypercolumn_size[0]), (j*hypercolumn_size[1]):((j+1)*hypercolumn_size[1])] = prefer_cluster_singles[i%2][j%2]
    nanpos = np.isnan(prefer_clusters)
    prefer_clusters[nanpos] = np.random.random(prefer_clusters.shape)[nanpos] * 360.0
    self.prefer_clusters = prefer_clusters
    self.iclusters = np.maximum(np.floor(prefer_clusters/delta_orientation), 0).astype(np.int32)
    self.cluster_num = cluster_num
    neuron_types = np.zeros(self.shape).astype(np.uint8) + T_INH
    random = np.random.random(self.shape)
    neuron_types[random < exc_ratio] = T_EXC
    self.neuron_types = neuron_types
    config.types = neuron_types
    # Neuron specifications
    config.types = neuron_types
    config.v_thre = config.fetch("v_thre", 1.0)
    config.v_reset = config.fetch("v_reset", 0)
    config.t_ref = config.fetch("t_ref", 0.01)
    if config.stimulus is None:
      from largescale.src.stimulus import DFStimulus
      sticonfig = CommonConfig()
      sticonfig.orientation = config.fetch("df_orientation", 0.0)
      sticonfig.frequency = config.fetch("df_frequency", 0.1)
      sticonfig.speed = config.fetch("df_speed", 5)
      config.stimulus = DFStimulus(self.shape, sticonfig)
    config.v_exc = config.fetch("v_exc", 4.67)
    config.v_inh = config.fetch("v_inh", -0.67)
    config.t_ref_exc = config.fetch("t_ref_exc", 0.003)
    config.t_ref_inh = config.fetch("t_ref_inh", 0.001)
    config.fgaba = config.fetch("fgaba", 0.0)
    config.fnmda = config.fetch("fnmda", 0.0)
    config.fgaba_noise = config.fetch("fgaba_noise", 0.0)
    config.fnmda_noise = config.fetch("fnmda_noise", 0.0)
    self.g_leak = config.fetch("g_leak", 50.0)
    self.firing_rate_noisy_exc_nmda = config.fetch("firing_rate_noisy_exc_nmda", 0.0)
    self.tau_rise_noisy_exc_nmda = config.fetch("tau_rise_noisy_exc_nmda", 0.0)
    self.tau_damp_noisy_exc_nmda = config.fetch("tau_damp_noisy_exc_nmda", 0.0)
    self.firing_rate_noisy_inh_gaba1 = config.fetch("firing_rate_noisy_inh_gaba1", 0.0)
    self.tau_rise_noisy_inh_gaba1 = config.fetch("tau_rise_noisy_inh_gaba1", 0.0)
    self.tau_damp_noisy_inh_gaba1 = config.fetch("tau_damp_noisy_inh_gaba1", 0.0)
    self.firing_rate_noisy_exc_ampa = config.fetch("firing_rate_noisy_exc_ampa", 0.0)
    self.tau_rise_noisy_exc_ampa = config.fetch("tau_rise_noisy_exc_ampa", 0.0)
    self.tau_damp_noisy_exc_ampa = config.fetch("tau_damp_noisy_exc_ampa", 0.0)
    self.firing_rate_noisy_inh_gaba2 = config.fetch("firing_rate_noisy_inh_gaba2", 0.0)
    self.tau_rise_noisy_inh_gaba2 = config.fetch("tau_rise_noisy_inh_gaba2", 0.0)
    self.tau_damp_noisy_inh_gaba2 = config.fetch("tau_damp_noisy_inh_gaba2", 0.00167)









