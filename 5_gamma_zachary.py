import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
from moduleNX import assign_partition, simulated_annealing
from networkx.algorithms.community import modularity


seed = 9
rn = np.random.default_rng(seed)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    'font.size': 14,            # Base size
    'axes.labelsize': 11,       # Axis labels
    'xtick.labelsize': 9,       # Axis ticks
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# ZACHARY NETWORK ------------------------------------------------------------------

gamma = 0.355

G_weight = nx.karate_club_graph()
G = nx.Graph()
G.add_nodes_from(G_weight.nodes(data=True))
G.add_edges_from(G_weight.edges())

N = len(G.nodes())

# Red 1: Zachary original
c1 = np.zeros(N)
G1 = G.copy()

# Red 2: Estructura óptica
G2 = G.copy()

# Structure in communities of G1
for ni,d in G1.nodes(data=True):
    if d['club']=='Mr. Hi': 
        c1[ni] = 0
        d['block'] = 0
    else:
        c1[ni] = 1
        d['block'] = 1

comm = [set(n for n,d in G1.nodes(data=True) if d['block']==i) for i in range(2)]
Q_Zach = modularity(G1, communities=comm, resolution=gamma)

# Structure in communities according to SA in G2
c_max, Q_max = simulated_annealing(G2, rn, T_random=5*N, p0=0.85, alpha=1.001, constant_T_iters=10*N, frozen_condition=N*10, gamma=gamma, print_vals=False)
assign_partition(G2, c_max)
comm = [set(n for n,d in G2.nodes(data=True) if d['block']==i) for i in range(len(np.unique(c_max)))]
Qf = modularity(G2, communities=comm, resolution=gamma)
print("Estos valores deberían coincidir:")
print(Q_max, r"Q tras sumar \delta Q's")
print(Qf, "Q recalculado con la c_max")

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,4), constrained_layout=True, dpi=150)
ax1:Axes
ax2:Axes
pos = nx.spring_layout(G, iterations=500, seed=seed, k=1/np.sqrt(N))

nodesize = 90
edgewidth = 1
xstep = 5
alpha_edges = 0.7

cmap = plt.get_cmap('tab10')
# colors1 = [cmap(d['block']) for _, d in G1.nodes(data=True)]
# Substitute your color definition block with this:
# Community 0: White with black border
# Community 1: Gray with white border
node_colors_ax1 = []
edge_colors_ax1 = []

for ni, d in G1.nodes(data=True):
    if d['block'] == 0:
        node_colors_ax1.append("white")
        edge_colors_ax1.append("black")
    else:
        node_colors_ax1.append("gray") # You can use a lighter gray with "#D3D3D3"
        edge_colors_ax1.append("black")

colors2 = [cmap(d['block']) for _, d in G2.nodes(data=True)]


# G1: REAL DIVISION IN 2 COMMUNITIES -------------------------------------------------

pos_labels = {}
# pos_labels[0] = (pos[0][0], pos[0][1] + 0.08)   # Mr. Hi arriba
# pos_labels[33] = (pos[33][0], pos[33][1] - 0.08) # Officer abajo
pos_labels[0] = (pos[0][0], pos[0][1]-0.005)   # 1
pos_labels[33] = (pos[33][0], pos[33][1]-0.005) # 2
lideres = [0, 33]
demas_nodos = [n for n in G.nodes() if n not in lideres]

# Colores normales
# nx.draw_networkx_nodes(
#     G1, pos, ax=ax1,
#     nodelist=demas_nodos,
#     node_color=[colors1[i] for i in demas_nodos],
#     node_shape='o', # Circle
#     node_size=nodesize,
#     edgecolors="black", linewidths=0.75
# )

# # Dibujar líderes (Cuadrados)
# nx.draw_networkx_nodes(
#     G1, pos, ax=ax1,
#     nodelist=lideres,
#     node_color=[colors1[i] for i in lideres],
#     node_shape='s', # 's' comes from Square
#     node_size=nodesize * 1.5, # Un poco más grandes para que resalten
#     edgecolors="black", linewidths=0.75
# )


nx.draw_networkx_edges(
    G1, pos, ax=ax1,
    edge_color="black",
    alpha=alpha_edges,
    width=edgewidth,
    style='solid',
)
# # nx.draw_networkx_labels(
# #     G1, ax=ax1,
# #     pos= pos_labels,
# #     labels={0: "Mr. Hi", 33: "Officer"},
# #     font_size=10,
# #     font_color="#000000FF",
# #     font_weight='bold',
# #     font_family='cm',
# #     verticalalignment='center',
# #     horizontalalignment='center',
# # )
# nx.draw_networkx_labels(
#     G1, ax=ax1,
#     pos= pos_labels,
#     labels={0: "1", 33: "2"},
#     font_size=10,
#     font_color="white",
#     font_weight='bold',
#     # font_family='cm',
#     verticalalignment='center',
#     horizontalalignment='center',
# )
# --------------------------------------------

# Colores blanco y negro

# Dibujar nodos normales (no líderes)
nx.draw_networkx_nodes(
    G1, pos, ax=ax1,
    nodelist=demas_nodos,
    node_color=[node_colors_ax1[i] for i in demas_nodos],
    node_shape='o',
    node_size=nodesize,
    edgecolors=[edge_colors_ax1[i] for i in demas_nodos], # Borde dinámico
    linewidths=0.9 # Aumentamos un poco el grosor del borde para que se note el blanco
)
# Dibujar líderes (Cuadrados)
nx.draw_networkx_nodes(
    G1, pos, ax=ax1,
    nodelist=lideres,
    node_color=[node_colors_ax1[i] for i in lideres],
    node_shape='s',
    node_size=nodesize * 1.5,
    edgecolors=[edge_colors_ax1[i] for i in lideres], # Borde dinámico
    linewidths=1.1
)
# Cambia el color de la fuente para que sea legible
nx.draw_networkx_labels(
    G1, ax=ax1,
    pos=pos_labels,
    labels={0: "1", 33: "2"},
    font_size=10,
    font_color="black", # Cambiado de "white" a "black" para que se vea en el nodo blanco
    font_weight='bold',
    verticalalignment='center',
    horizontalalignment='center',
)



ax1.set_title("(a) Zachary: " + f"$Q={Q_Zach:.3f}$")
ax1.set_axis_off()


# G2: DIVISION ACCORDING TO SA -----------------------------------------------------------
nx.draw_networkx_nodes(
    G2, pos, ax=ax2,
    nodelist=demas_nodos,
    node_color=[colors2[i] for i in demas_nodos],
    node_shape='o', # Circle
    node_size=nodesize,
    edgecolors="black", linewidths=0.9
)
# Dibujar líderes (Cuadrados)
nx.draw_networkx_nodes(
    G2, pos, ax=ax2,
    nodelist=lideres,
    node_color=[colors2[i] for i in lideres],
    node_shape='s', # 's' comes from Square
    node_size=nodesize * 1.5, # Un poco más grandes para que resalten
    edgecolors="black", linewidths=1.1
)

## edges
nx.draw_networkx_edges(
    G2, pos, ax=ax2,
    edge_color="black",
    alpha=alpha_edges,
    width=edgewidth,
    style='solid',
)

ax2.set_title("(b) SA: " + fr"$Q={Q_max:.3f}$  " + fr"$\gamma={gamma:.3f}$")
ax2.set_axis_off()


# plt.savefig('zachary.pdf', dpi=300, bbox_inches="tight")
plt.show()