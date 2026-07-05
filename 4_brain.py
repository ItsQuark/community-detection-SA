import numpy as np
import pandas as pd
import scipy.io
import networkx as nx
from moduleNX import plot_network, simulated_annealing, assign_partition
import time

seed = None
rn = np.random.default_rng(seed)

data = scipy.io.loadmat('data/brain/SCmatrices88healthy.mat')
matrices = data['SCmatrices']

data_roi = pd.read_csv('data/brain/AAL_regions.csv', delimiter=';')
roiID = data_roi['ROI number'].to_numpy()-1
roiName = data_roi['ROI name'].to_numpy()

# a = Average weighted adjacency matrix of the 88 subjects
a = np.mean(matrices, axis=0)
N = np.shape(a)[0]

# Clean the diagonal
np.fill_diagonal(a, 0)

# b = Transformation of 'a' to unweighted matrix by applying threshold u: a[i,j]>=thr then a[i,j]=1, and 0 otherwise
thr = 0.01            
b = (a>=thr).astype(int)

# Graph from adjacency matrix b
G = nx.from_numpy_array(b)

for n,d in G.nodes(data=True):
    d['roi'] = roiName[n]
# plot_network(G, layout='fa2', fa2_iters=200)

# Gamma = 2.18 gives 8 communities robustly

for gamma in [2.18]:
    
    start = time.perf_counter()
    cmax, Qmax = simulated_annealing(G, rn, T_random=5*N, p0=0.85, frozen_condition=50*N, alpha=1.005, constant_T_iters=10*N, gamma=gamma, betamax=1e9)
    end = time.perf_counter()
    num_com = len(np.unique(cmax))
    assign_partition(G, cmax)

    print(Qmax, num_com)

    plot_network(G, Q=Qmax, layout='fa2', fa2_iters=200,axis_ticks=np.linspace(0,89,4), colormap='tab10',
                figsize=(8.5,4.5), axislabel_size=13, axisticks_size=13, nodesize=90, edgewidth=0.7, alpha_edges=0.9,
                # filename=f'brain_{gamma:.2f}_{Qmax:.3f}'
                )

    for i in range(num_com):
        l = []
        for n,d in G.nodes(data=True):
            if d['block']==i: l.append(d['roi'])
        print(f"Community {i+1} ({len(l)} partes): {l}")

# --------------------------------------------------------------------------- END


# ------------------------------------------------------------------------------------------------------------------ robustness

# n_runs = 5
# cmaxs = []
# roi_sets = []  # list of lists of frozensets (one per run)

# for run in range(n_runs):
#     rn = np.random.default_rng()
#     cmax_i, Qmax_i = simulated_annealing(G, rn, T_random=5*N, p0=0.85,
#                                           frozen_condition=10*N, alpha=1.005,
#                                           constant_T_iters=10*N, betamax=1e9, gamma=gamma)
#     cmaxs.append(cmax_i)
    
#     # Save composition of each community as a frozenset of ROI names
#     assign_partition(G, cmax_i)
# communities = {}
#     for n, d in G.nodes(data=True):
#         c = d['block']
#         if c not in comunidades:
#             comunidades[c] = set()
#         comunidades[c].add(roiName[n])
#     roi_sets.append([frozenset(s) for s in comunidades.values()])
    
#     print(f"Run {run+1}: Q={Qmax_i:.4f}, communities={len(np.unique(cmax_i))}")

# # Compare each run with the first as reference
# ref = roi_sets[0]
# print("\nComparison against Run 1:")
# for run in range(1, n_runs):
#     total_diff = 0
#     for com_ref in ref:
#         # Find the most similar community in this run
#         mejor_match = max(roi_sets[run], key=lambda s: len(s & com_ref))
#         diff = len(com_ref.symmetric_difference(mejor_match))
#         total_diff += diff
#     print(f"  Run {run+1}: {total_diff} ROIs in different communities")

# -----------------------------------------------------------------------------------------------------------------

pruebas = 100
nf=8
num_com = np.zeros(pruebas)
Q = np.zeros(pruebas)

continua = True
gamma = 2.24
i = 0

cont = 0
for i in range(pruebas):    
    start = time.perf_counter()
    cmax, Qmax = simulated_annealing(G, rn, T_random=5*N, p0=0.85, frozen_condition=10*N, alpha=1.005, constant_T_iters=10*N, gamma=gamma, betamax=1e9)
    end = time.perf_counter()
    num_com[i] = len(np.unique(cmax))
    Q[i] = Qmax
    print(f'Q = {Qmax:.4f} | t = {end-start} seg | num = {num_com[i]}')
    if num_com[i]!=nf:
        cont+=1
        print("Distintos de 8 detectados:", cont)
    

print(f"numero de comunidades = {np.mean(num_com)} +- {np.std(num_com)}\n")
print(f"{Q}")
print(f"Q = {np.mean(Q):.4f} +- {np.std(num_com)}")