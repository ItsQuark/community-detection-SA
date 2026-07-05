import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import networkx as nx
from networkx.algorithms.community import girvan_newman, modularity
from moduleNX import assign_partition
from scipy.optimize import linear_sum_assignment

seed = None
rn = np.random.default_rng(seed)
# rn_warmup = np.random.default_rng(0)
# numba_warm_up(rn_warmup)

plt.rcParams.update({
    'font.size': 11,            # Base size
    'axes.labelsize': 11,       # Axis labels
    'xtick.labelsize': 9,       # Axis ticks (numbers)
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# -----------------------------------------------------------------------------

# Function that sorts the graph output of the SA algorithm

def sort_graph(Gf:nx.Graph, c_max:np.ndarray, N:int)->tuple[nx.Graph, np.ndarray]:
    assign_partition(G=Gf, c=c_max)
    idx = np.argsort(c_max)
    c_max_ordered = np.zeros(N,np.int32)
    c_max_ordered[:] = c_max[idx]

    remap = {old:new for new, old in enumerate(idx)}
    Gaux = nx.relabel_nodes(Gf, remap)
    Gremap = nx.Graph()
    Gremap.add_nodes_from(sorted(Gaux.nodes()))
    Gremap.add_edges_from(Gaux.edges())

    for n,d in Gremap.nodes(data=True):
        d['block']=c_max_ordered[n]
    
    return Gremap, c_max_ordered

# Function that finds the fraction of correctly classified nodes

def classify(ci: np.ndarray, c_pred: np.ndarray, N: int) -> float:
    labels_true = np.unique(ci)
    labels_pred = np.unique(c_pred)
    n = max(len(labels_true), len(labels_pred))
    
    # Confusion matrix
    cost = np.zeros((n, n), dtype=np.int32)
    for j, lt in enumerate(labels_true):
        for k, lp in enumerate(labels_pred):
            cost[j, k] = np.sum((ci == lt) & (c_pred == lp))
    
    # Optimal label assignment
    row_ind, col_ind = linear_sum_assignment(-cost)
    return cost[row_ind, col_ind].sum() / N

def gn_best_partition(G: nx.Graph) -> np.ndarray:
    """
    Iterates Girvan-Newman and returns the label array
    corresponding to the partition with maximum modularity.
    Does not need to know the number of communities.
    """
    N = G.number_of_nodes()
    best_q = -np.inf
    best_c = np.zeros(N, dtype=np.int32)

    for partition in girvan_newman(G):
        q = modularity(G, partition)
        if q > best_q:
            best_q = q
            c = np.zeros(N, dtype=np.int32)
            for label, community in enumerate(partition):
                for node in community:
                    c[node] = label
            best_c = c
        elif q < best_q - 0.05:
            # Early stop: modularity has been falling for a while
            break

    return best_c

# -----------------------------------------------------------------------------

k = 16

# a = 0.625   # Max k_out/k that I will represent
# vals = 21   # number of points
# nprom = 30  # number of data to average

# a = 0.625     # Max k_out/k that I will represent
# vals = 4    # number of points
# nprom = 5

# a = 0.625
# vals = 2
# nprom = 1

a = 0.78125
vals = 26
nprom = 40

k_out = np.linspace(0, a*k, num=vals, endpoint=True, dtype=np.float32)
x = k_out/k
y_sa = np.zeros_like(k_out)
y_gn = np.zeros_like(k_out)
y_sa_prom = np.zeros_like(k_out)
y_gn_prom = np.zeros_like(k_out)

N = 32+32+32+32

# for iter in range(nprom):
    
#     for i, k_out_i in enumerate(k_out):
#         Gi, Qi = sbm_gn([32,32,32,32], k=k, k_out=k_out_i, seed=seed)

#         # True communities
#         ci = np.array([d['block'] for _,d in Gi.nodes(data=True)], np.int32)
         
#         # CHANGE: save mapping instead of ignoring it
#         Gf, mapping = shuffled_graph(Gi, rn=rn, shuffle=True)

#         # True labels aligned with the shuffled indices of Gf
#         ci_shuffled = np.zeros(N, dtype=np.int32)
#         for orig, shuf in mapping.items():
#             ci_shuffled[shuf] = ci[orig]

#         # SA
#         Gf_SA = Gf.copy()
        
        
#         t0 = time.perf_counter()
#         c_max, Q_max = simulated_annealing(Gf_SA, rn, T_random=200, p0=0.85, alpha=1.001, frozen_condition=10*N)
#         print(f"[SA] Network {i+1} — {time.perf_counter()-t0:.3f} s")
#         y_sa[i] = classify(ci_shuffled, c_max, N)

#         # GN
#         t0 = time.perf_counter()
#         c_gn = gn_best_partition(Gf)
#         print(f"[GN] Network {i+1} — {time.perf_counter()-t0:.3f} s")
#         y_gn[i] = classify(ci_shuffled, c_gn, N)
        
    
#     print(f"Loop {iter} completed.")
#     print(f"y_sa = {y_sa}")
#     print(f"y_gn = {y_gn}")
    
#     # Promedio
#     y_sa_prom += y_sa
#     y_gn_prom += y_gn

# y_sa_prom = y_sa_prom/nprom
# y_gn_prom = y_gn_prom/nprom

# # Displaying data
# for i in range(len(x)):
#     if i==0: 
#         print("x\t y(SA)\t        y(GN)")
#     print(f"{x[i]:.3f}\t {y_sa_prom[i]:.7f}\t {y_gn_prom[i]:.7f}")
    
# # Storing data
# np.savez('guimera.npz', y_sa_prom=y_sa_prom, y_gn_prom=y_gn_prom)

# Reading data
data = np.load('guimera40sim.npz')
y_sa_prom_JM = data['y_sa_prom']
y_gn_prom_JM = data['y_gn_prom']

data = np.load('guimera40_AD.npz')
y_sa_prom_AD = data['y_sa_prom']
y_gn_prom_AD = data['y_gn_prom']

# print("x \t y(SA) \t y(GN)")
# for xi,ysai, ygni in zip(x,y_sa_prom_AD,y_gn_prom_AD):
#     print(f"{xi:.5f} \t {ysai:.5f}  {ygni:.5f}")
# for xi,ysai, ygni in zip(x,y_sa_prom_JM,y_gn_prom_JM):
#     print(f"{xi:.5f} \t {ysai:.5f}  {ygni:.5f}")

y_sa_prom=0.5*(y_sa_prom_JM+y_sa_prom_AD)
y_gn_prom=0.5*(y_gn_prom_JM+y_gn_prom_AD)

print(r"\toprule")
print(r"$k_{out}/k$ & Fracción SA & Fracción GN \\")
print(r"\midrule")
for xi,ysai, ygni in zip(x,y_sa_prom,y_gn_prom):
    print(f"{xi:.5f} & {ysai:.4f} & {ygni:.4f}"+r"\\")
print(r"\bottomrule")

fig, ax = plt.subplots(figsize=(4.5,4), constrained_layout=True, dpi=130)
ax.plot(x, y_sa_prom, color='black', marker='s', markersize=6, linewidth=0.7, label="Simulated Annealing")
ax.plot(x, y_gn_prom, color='black', marker='o', markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.65, markersize=7, linewidth=0.7, label="Girvan-Newman")

for li in [1,5,8]:
    ax.axvline(x=li/k, color='black', linestyle='--', linewidth=0.8, alpha=0.9, zorder=-7)    

ax.set_xlabel(r'Fracción de aristas intercomunitarias, $k_{out}/k$')
ax.set_ylabel(r'Fracción correctamente clasificada')
ax.set_ylim(0.0001, 1.05)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.tick_params(which='both', direction='in', top=True, right=True)

ax.grid(which='major', linestyle='--', alpha=0.5, zorder=-10)
ax.grid(which='minor', linestyle='--', alpha=0.4, zorder=-10)
ax.legend(loc="lower left", frameon=True, edgecolor='none', facecolor='white', framealpha=0.7)

plt.savefig('comparativa_sa_gn_80sim.pdf', dpi=300, bbox_inches="tight")
plt.show()