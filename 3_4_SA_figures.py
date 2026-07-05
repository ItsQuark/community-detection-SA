import numpy as np
from moduleNX import sbm_gn, plot_network

seed = None
rn = np.random.default_rng(seed)

k = 16
k_out = np.array([1,5,8])
x = k_out/k

N = 32+32+32+32

for i, k_out_i in enumerate(k_out):
    Gi, Qi = sbm_gn([32,32,32,32], k=k, k_out=k_out_i, seed=seed)
    plot_network(
        Gi, Qi, show_net=True, colormap='tab10', axis_ticks=np.linspace(0, N-1, 5), label_pad=-25, figsize=(12,6), axislabel_size=13, axisticks_size=12
        # filename=f'planted{k_out_i}'
        )