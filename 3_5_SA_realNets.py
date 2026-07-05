import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
from moduleNX import plot_network, assign_partition, simulated_annealing, numba_warm_up

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

seed = 9
rn = np.random.default_rng(seed)
rn_warmup = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    'font.size': 14,            # Base size
    'axes.labelsize': 11,       # Axis labels
    'xtick.labelsize': 9,       # Axis ticks (numbers)
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# DOLPHINS ---------------------------------------------------------------------------------------------------
""" The file dolphins.gml contains an undirected social network of frequent
    associations between 62 dolphins in a community living off Doubtful Sound,
    New Zealand, as compiled by Lusseau et al. (2003).  Please cite

    D. Lusseau, K. Schneider, O. J. Boisseau, P. Haase, E. Slooten, and
    S. M. Dawson, The bottlenose dolphin community of Doubtful Sound features
    a large proportion of long-lasting associations, Behavioral Ecology and
    Sociobiology 54, 396-405 (2003).

    Additional information on the network can be found in

    D. Lusseau, The emergent properties of a dolphin social network,
    Proc. R. Soc. London B (suppl.) 270, S186-S188 (2003).

    D. Lusseau, Evidence for social role in a dolphin social network,
    Preprint q-bio/0607048 (http://arxiv.org/abs/q-bio.PE/0607048)
"""

if False:

    G = nx.read_gml(path="data/dolphins/dolphins.gml")
    G = nx.convert_node_labels_to_integers(G)  # nodes become 0...N-1
    N = len(list(G.nodes()))

    start = time.perf_counter()
    c_max, Q_max = simulated_annealing(G, rn, T_random=N*500, p0=0.9, constant_T_iters=10*N, frozen_condition=N*10, alpha=1.0008, print_vals=True, betamax=1e8)
    print(f"Q = {Q_max:.4f} \t {time.perf_counter()-start:.4f} s")

    assign_partition(G, c_max)

    Gremap, c_max_ordered = sort_graph(G, c_max, N=N)
    plot_network(G=G, Q=Q_max, nodesize=60, nodeborderwidth=0.75, edgewidth=0.5, alpha_edges=0.7, layout='fa2', fa2_iters=100, show_matrix=False, colormap='tab10')

# FOOTBALL ---------------------------------------------------------------------------------------------------
""" The file football.gml contains the network of American football games
    between Division IA colleges during regular season Fall 2000, as compiled
    by M. Girvan and M. Newman. The nodes have values that indicate to which
    conferences they belong. The values are as follows:

    0 = Atlantic Coast
    1 = Big East
    2 = Big Ten
    3 = Big Twelve
    4 = Conference USA
    5 = Independents
    6 = Mid-American
    7 = Mountain West
    8 = Pacific Ten
    9 = Southeastern
    10 = Sun Belt
    11 = Western Athletic

    If you make use of these data, please cite M. Girvan and M. E. J. Newman,
    Community structure in social and biological networks,
    Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002).
"""

if False:

    G = nx.read_gml(path="data/football/football.gml")
    G = nx.convert_node_labels_to_integers(G)  # nodes become 0..N-1
    N = len(list(G.nodes()))

    start = time.perf_counter()
    c_max, Q_max = simulated_annealing(G, rn, T_random=200, p0=0.85, frozen_condition=N*10, alpha=1.0005)
    print(f"Q = {Q_max:.4f} \t {time.perf_counter()-start:.4f} s")

    assign_partition(G, c_max)

    Gremap, c_max_ordered = sort_graph(G, c_max, N=N)
    plot_network(Gremap, 50, 0.7, 20, 0.5, Q=Q_max)

# POWER GRID --------------------------------------------------------------------------------------------------
"""Western States Power Grid

    Compiled by Duncan Watts and Steven Strogatz

    The file power.gml contains an undirected unweighted representation of the
    topology of the Western States Power Grid of the United States, compiled by
    Duncan Watts and Steven Strogatz.  The data are from the web site of
    Prof. Duncan Watts at Columbia University,
    http://cdg.columbia.edu/cdg/datasets.  Node IDs are the same as those used
    by Prof. Watts.

    These data can be cited as:

    D. J. Watts and S. H. Strogatz, "Collective dynamics of `small-world'
    networks", Nature 393, 440-442 (1998).
"""
   
if True:

    G = nx.read_gml(path="data/power/power.gml", label='id')
    G = nx.convert_node_labels_to_integers(G)  # nodes become 0..N-1
    N = len(list(G.nodes()))

    # If False, run 'simulated_annealing' to create the file 'filename_data.npz'
    # If True, uses 'filename_data.npz' to generate 'filename_plot.pdf'
    existing_data = True
    dataset = 4
    
    if dataset==1:
        # 643 communities found in 6 min, Mean size: 7.672 | Standard deviation: 4.366
        filename_data = 'power_grid'
        filename_plot = 'power_grid_1'
    elif dataset==2:
        # Config: T_random=N*500, p0=0.9, constant_T_iters=3*N, frozen_condition=N*10, alpha=1.0008, betamax=1e8
        # Result: 402 communities found in 18.9 min, Mean size: 12.261 | Standard deviation: 7.919. last beta reached=1.3e7 < betamax, OK, Q_max=0.822
        filename_data = 'power_grid_2'
        filename_plot = 'power_grid_2'
    elif dataset==3:
        # Config: T_random=N*500, p0=0.9, constant_T_iters=10*N, frozen_condition=N*10, alpha=1.0005, betamax=1e8
        # Result: 216 communities found in 80 min, Mean size: 22.770 | Standard deviation: 17.685 .last beta reached=1.9e6 < betamax, OK, Q_max=0.878
        filename_data = 'power_grid_3'
        filename_plot = 'power_grid_3'
    elif dataset==4:
        # Config: T_random=N*500, p0=0.9, constant_T_iters=, frozen_condition=N*10, alpha=, betamax=1e8
        # Result:  communities found in  min, Mean size:  | Standard deviation:  .last beta reached= < betamax, OK, Q_max=
        filename_data = 'power_grid_'
        filename_plot = 'power_grid_'
    elif dataset==5:
        # Config: T_random=N*500, p0=0.9, constant_T_iters=, frozen_condition=N*10, alpha=, betamax=1e8
        # Result:  communities found in  min, Mean size:  | Standard deviation:  .last beta reached= < betamax, OK, Q_max=
        filename_data = 'power_grid_'
        filename_plot = 'power_grid_'
    else:
        # Default name just in case
        filename_data = 'default_data'
        filename_plot = 'default_plot'
    
    if not existing_data:
        numba_warm_up(rn_warmup)
        
        start = time.perf_counter()
        c_max, Q_max = simulated_annealing(G, rn, T_random=N*500, p0=0.9, constant_T_iters=10*N, frozen_condition=10*N, alpha=1.0005, print_vals=True, betamax=1e8)
        print(f"Q = {Q_max:.4f} \t {time.perf_counter()-start:.4f} s")
        np.savez(filename_data+'.npz', c_max=c_max, Q_max=Q_max)
    else:
        data = np.load(filename_data+'.npz')
        c_max = data['c_max']
        Q_max = data['Q_max']
        
        com_id, size = np.unique(c_max, return_counts=True)
        mean_nodes = np.mean(size)
        std_nodes = np.std(size)
        for v, o in zip(com_id, size):
            print(f"Community {v}: {o} nodes")
        print(f"Mean size: {mean_nodes:.3f} | Standard deviation: {std_nodes:.3f}")
    
    # Plot
    assign_partition(G, c_max)
    plot_network(G=G, Q=Q_max, nodesize=10, nodeborderwidth=0.4, edgewidth=0.5, alpha_edges=0.7, layout='fa2', fa2_iters=500, filename=filename_plot, show_matrix=False, colormap='turbo', seed=9)