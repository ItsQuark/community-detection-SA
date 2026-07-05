import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter
from matplotlib.axes import Axes
import networkx as nx
from networkx.algorithms.community import modularity
from fa2 import ForceAtlas2
from numba import njit
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", 'font.size': 14})

def sbm_gn(community_structure:list, k:float, k_out:float, seed:int|None=None) -> tuple[nx.Graph, float]:
    
    # Redefine with convenient names and data structures
    nr = np.array(community_structure)
    ko = k_out
    
    r = len(nr)
    n = nr[0]   # Read important restriction (a)
    N = n*r
    
    # Read imp rest (b)
    po = ko/(n*(r-1))
    pi = (k-ko)/(n-1)

    P = np.full((r,r), po)
    np.fill_diagonal(P, pi)

    G = nx.stochastic_block_model(sizes=nr, p=P, seed=seed)

    # Ex. of G.nodes(data=True): 
    # [(0, {'block': 0}), (1, {'block': 0}), (2, {'block': 0}), (3, {'block': 1}), (4, {'block': 1}), (5, {'block': 1}), (6, {'block': 1}), (7, {'block': 1})]
    
    comm = [set(n for n,d in G.nodes(data=True) if d['block']==i) for i in range(r)]
    
    # Modularity
    Q = modularity(G, comm)
    
    return G,Q
    
def plot_network(G:nx.Graph, Q:float|None=None, nodesize:float|None=90, nodeborderwidth:float|None=0.75, edgewidth:float|None=0.9, xstep:int|None=None, alpha_edges:float|None=0.9, with_labels:bool|None=None, layout:str|None=None, filename:str|None=None, show_matrix:bool|None=None, show_net:bool|None=None, colormap:str|None='tab10', spring_iters:int|None=None, fa2_iters:int|None=None, fa2_verbose:bool=False, seed:int|None=None, axis_ticks:np.ndarray|None=None, set_title:bool|None=False, figsize:tuple[float,float]|None=(10,5), axislabel_size:float|None=14, axisticks_size:float|None=14, label_pad:float|None=-17) -> None:
    """
    Args:
        axis_ticks (np.ndarray): if 90 nodes are passed, a matrix is represented with rows and cols from 0 to 89. I want to have two intermediate points, the first one at 0+x, such that 0+4x=89, x=89//4, then I make axis_ticks=np.array([0, x, 2x, 3x, 4x=89]) or axis_ticks=np.linspace(0,89,4). Then, internally the function labels tick t as t+1, so that instead of showing 0, it shows 1, and instead of showing 89, it shows 90.
        label_pad (float|None=-17): separation of the y-axis title.
    """
    
    N = len(G.nodes())
    
    # Check optional arguments
    if xstep is None and axis_ticks is None:
        xstep = N//4
    
    # Calculate 'pos' variable
    if layout is None or layout == 'spring':
        iters = spring_iters if spring_iters is not None else 500
        pos = nx.spring_layout(G, iterations=iters)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'fa2':
        iters = fa2_iters if fa2_iters is not None else 100
        fa2 = ForceAtlas2(verbose=fa2_verbose, seed=seed)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=iters)
    else:
        raise ValueError(f"Layout '{layout}' no reconocido. Opciones: 'spring', 'spectral', 'kamada_kawai', 'fa2'")

    # Check whether nodes have communities assigned in the dictionary in G.nodes(data=True)
    no_communities_assigned = sum([len(d) for n,d in G.nodes(data=True)])==0
    if no_communities_assigned:
        # Simple plot
        nx.draw(G, pos=pos, with_labels=(with_labels is True), node_size=nodesize or 90)
        plt.tight_layout()
        plt.show()
        if filename is not None:
            plt.savefig(filename+'.pdf', dpi=300, bbox_inches="tight")
        return

    # What to plot
    if ((show_matrix is None) or (show_matrix==True)) and (show_net is None or show_net==True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, dpi=130, gridspec_kw={'width_ratios': [2, 1.2]})
    elif (show_net==None or show_net==False) and (show_matrix is None or show_matrix==True):
        fig, ax2 = plt.subplots(figsize=figsize, constrained_layout=True, dpi=130)
    elif show_matrix==False:
        fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True, dpi=130)
    else:
        print("Why have you called me then?")
        return
    ax1:Axes
    ax2:Axes
    
    # Colormap
    cmap = plt.get_cmap(colormap)
    if colormap=='turbo' or colormap=='viridis' or colormap=='inferno':
        # 1. Get the maximum number of communities
        num_comunidades = max(d['block'] for _, d in G.nodes(data=True)) + 1
        # 2. Create the list of colors by normalizing the block ID
        # d['block'] / num_comunidades ensures the value is between 0 and 1
        colors = [cmap(d['block'] / num_comunidades) for n, d in G.nodes(data=True)]
    elif colormap=='tab10' or colormap=='tab20':
        colors = [cmap(d['block']) for n, d in G.nodes(data=True)]
    else:
        raise ValueError(f"colormap '{colormap}' no reconocido. Opciones: 'tab10', 'tab20', 'turbo', 'viridis', 'inferno'")
    
    if (((show_matrix is None) or (show_matrix==True)) and (show_net is None or show_net==True)) or show_matrix==False:
        
        ## nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax1,
            node_color=colors,
            edgecolors="black",
            linewidths=nodeborderwidth or 0.75,
            node_size=nodesize or 90,
            node_shape='o',
            alpha=1.0,
        )
        ## edges
        nx.draw_networkx_edges(
            G, pos, ax=ax1,
            edge_color="black",
            alpha=alpha_edges or 0.9,
            width=edgewidth or 0.9,
            style='solid',
        )
        
        if with_labels is not None:
            if with_labels==True:    
                # labels
                nx.draw_networkx_labels(
                    G, pos, ax=ax1,
                    labels=None,                     # Dictionary {node: text}; None uses the node name
                    font_size=7,                     # Font size
                    font_color="#000000CF",        # Font color
                    font_weight='normal',            # Weight: 'normal', 'bold', 'light'
                    font_family='sans-serif',        # Typography
                    verticalalignment='center',      # Vertical alignment of the text
                    horizontalalignment='center',    # Horizontal alignment of the text
                )

        
        if set_title==True: ax1.set_title("Modularidad: " + f"$Q={Q:.3f}$")
        ax1.set_axis_off()
    
    # Adjacency matrix visualization (only for small networks)
    if show_matrix is None or show_matrix==True:
        if len(G.nodes()) <= 1000:
            A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            im = ax2.imshow(A, cmap='binary')
            # Adjacency matrix visualization
            A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            im = ax2.imshow(A, cmap='binary')

            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes("right", size="4%", pad=0.15)
            # fig.colorbar(im, cax=cax)
            
            if set_title==True: ax2.set_title('Adjacency Matrix', y=1.18, fontsize=10)
            ax2.set_ylabel('Nodos', loc='top', labelpad=label_pad, fontsize=axislabel_size)
            ax2.xaxis.tick_top()
            
            if xstep is not None and axis_ticks is None:
                def offset_labels(x, pos):
                    return f'{int(x + 1)}'
                ax2.xaxis.set_major_locator(MultipleLocator(xstep))
                ax2.xaxis.set_major_formatter(FuncFormatter(offset_labels))
                ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax2.yaxis.set_major_locator(MultipleLocator(xstep))
                ax2.yaxis.set_major_formatter(FuncFormatter(offset_labels))
                ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
            elif xstep is None and axis_ticks is not None:
                axis_ticks = axis_ticks.astype(int)
                axis_ticks_labels = [str(i+1) for i in axis_ticks]
                ax2.set_xticks(axis_ticks, axis_ticks_labels)
                ax2.set_yticks(axis_ticks[1:], axis_ticks_labels[1:])
                
            ax2.tick_params(axis='both', labelsize=axisticks_size)
            
            # # Show modularity
            # if Q is not None: 
            #     ax1.text(0.1, 0.1, f"$Q={Q:.3f}$", 
            #             transform=ax1.transAxes, 
            #             ha='center', va='top', 
            #             fontsize=20,
            #     )
        else:
            ax2.text(0.5, 0.5, "Red demasiado grande\npara visualizar la\nmatriz de adyacencia",
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_axis_off()
    
    if filename is not None:
        plt.savefig(filename+'.pdf', dpi=300, bbox_inches="tight")
    plt.show()

def shuffle_graph(G, rn, shuffle:bool):
    mapping = {}
    nodes_list = list(G.nodes())
    nodes_list2 = nodes_list.copy()
    
    if shuffle: rn.shuffle(nodes_list2)
    
    for nodeID in nodes_list:
        mapping[nodeID] = nodes_list2[nodeID]
    return nx.relabel_nodes(G, mapping), mapping

def shuffled_graph(G, rn, shuffle:bool):
    G_aux, mapping = shuffle_graph(G, rn=rn, shuffle=shuffle)
    Gf = nx.Graph()
    Gf.add_nodes_from(sorted(G_aux.nodes()))
    Gf.add_edges_from(G_aux.edges())
    return Gf, mapping

@njit
def shift(c:np.ndarray, occur:np.ndarray, Kinn:np.ndarray, mi:int, N:int):
    """
    Changes the following arrays: c, occur, Kinn.
    
    c: it has the same effect as c[c>mi]-=1, but the code is accomodated for njit compilation.
    occur: it shifts to the left all elements that go after a hole.
    Kinn: same as occur.
    
    ### Example: 
        c = [0 0 1 0 1 2 1 3! 4 5 0 6] -(3 changes to 0)-> [0 0 1 0 1 2 1 0! 3 4 0 5]
    occur = [4 3 1 1! 1 1 1 0 0 0 0 0] -(occurrence of 3 is 0, but rearrangement to the left)-> [5 3 1 1! 1 1 0 0 0 0 0 0]
    """
    for i in range(N):
        if c[i]>mi: 
            c[i]-=1
    for i in range(mi,N-1):
        occur[i]=occur[i+1]
        Kinn[i]=Kinn[i+1]
    
def assign_partition(G, c:np.ndarray):
    """Completes the attribute of G of the NetworkX class that shows the community of node n.
    """
    for n,d in G.nodes(data=True):
        d['block'] = c[n]

@njit
def calculate_inner_degrees(c:np.ndarray, k:np.ndarray, N:int)->np.ndarray:
    """IMP: Kinn_zeros must be the desired array that will be returned but initiallized with zeros
    """
    Kinn_zeros = np.zeros(N,np.int64)
    for i in range(N):
        Kinn_zeros[c[i]] += k[i]
    return Kinn_zeros

@njit
def extract_neighbours(A_indices:np.ndarray, A_indptr:np.ndarray, k_to_mi_mj:np.ndarray, l:int, c:np.ndarray, mi:int, mj:int)->np.ndarray:
    """We could use the adjacency matrix, but it would be terribly inneficient since most of the memory (if storagble in RAM, which I'm afraid not for N~1e5 nodes) would be occupied with 0's (most of the nodes don't have so many links). This is a sparse matrix, in which only a small percent of data is informative. Instead, we use a compression method called 'Compressed Sparse Row'. Details on pg. 10 in my iPad. In short (Gemini):
    
    1. data (or A_data): Stores the values of the cells that are not zero. In an unweighted graph, this array is not even needed because all links have a value of 1.
    2. indices (or A_indices / col_ind): Stores the indices of the columns of those elements. In a graph, these are, literally, the IDs of the neighbors.
    3. indptr (or A_indptr / row_ptr): Stands for Index Pointer. It is a roadmap of size $N+1$ that tells you at which position in the array indices each row (each node) begins and ends.

    Args:
        A_indices (np.ndarray): A_indices
        A_indptr (np.ndarray): A_indptr
        k_to_mi_mj (np.ndarray): array of dim 2; k_to_mi_mj[0] gives the number of links from node l to community mi. k_to_mi_mj[1] gives the number of links from node l to community mj. It may or not be initialized to [0,0], since I will do it anyways.
        l (int): node ID in which neighbours we are interested.
        c (np.ndarray): array of community labels
        mi (int): k_to_mi_mj[0] gives the number of links from node l to community mi.
        mj (int): k_to_mi_mj[1] gives the number of links from node l to community mj.

    Returns:
        np.ndarray: updated k_to_mi_mj
    """
    # Retrieving the neighbors of node l is now super easy, it is achieved like this:
    # Vecinos del nodo l = A_indices[A_indptr[l]:A_indptr[l+1]], con l=0,1,...,N-1
    neighbours_of_l = A_indices[A_indptr[l]:A_indptr[l+1]]
    
    k_to_mi_mj[:]=0
    for neighbourID in neighbours_of_l:
        if c[neighbourID]==mi:
            k_to_mi_mj[0]+=1
        elif c[neighbourID]==mj:
            k_to_mi_mj[1]+=1
        
    return k_to_mi_mj

@njit
def delta_Q_gamma(gamma, K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn):
    """Calculate $\\Delta Q$ with gamma factor

    Args:
        gamma (_type_): gamma factor
        K_inv (_type_): 1/K, K=number of edges
        K2K_inv (_type_): 1/(2*K*K)
        k (_type_): array of degrees
        l (_type_): _description_
        mi (_type_): _description_
        mj (_type_): _description_
        k_to_mi_mj (_type_): _description_
        Kinn (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (k_to_mi_mj[1]-k_to_mi_mj[0])*K_inv - gamma*k[l]*(k[l]+Kinn[mj]-Kinn[mi])*K2K_inv

@njit
def delta_Q(K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn):
    """Calculate $\\Delta Q$

    Args:
        K_inv (_type_): 1/K, K=number of edges
        K2K_inv (_type_): 1/(2*K*K)
        k (_type_): array of degrees
        l (_type_): _description_
        mi (_type_): _description_
        mj (_type_): _description_
        k_to_mi_mj (_type_): _description_
        Kinn (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (k_to_mi_mj[1]-k_to_mi_mj[0])*K_inv - k[l]*(k[l]+Kinn[mj]-Kinn[mi])*K2K_inv

def SA_randwalk(T_random, p0, N, c, m_empty, occur, A_indices, A_indptr, K_inv, K2K_inv, k, Kinn, k_to_mi_mj, rn, gamma:float|None=None):
    
    bad_deltaQ = 0.0
    cont = 0
    
    for _ in range(T_random):
        
        # Choose randomly both a node that belongs to some community mi, as well as some mj!=mi
        l = rn.integers(0,N-1,endpoint=True)
        mi = c[l]
        mj = mi
        while mj==mi:
            mj = rn.integers(0, m_empty, endpoint=True)

        if mj==m_empty:         # All proposals are accepted here
            if occur[mi]==1:    # Condition (1)
                mj=mi
            elif occur[mi]>1:   # Condition (2)
                
                # Calculate deltaQ
                k_to_mi_mj = extract_neighbours(A_indices, A_indptr, k_to_mi_mj, l, c, mi, mj)
                if gamma is None: 
                    deltaQ = delta_Q(K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)
                else:
                    deltaQ = delta_Q_gamma(gamma, K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)    
                
                if deltaQ<0: 
                    bad_deltaQ += deltaQ
                    cont += 1
                
                # if rn.random() < np.min(1, np.exp(beta*deltaQ)):
                occur[mi]-=1
                occur[mj]+=1
                m_empty+=1
                
                Kinn[mi]-=k[l]
                Kinn[mj]+=k[l]

                c[l]=mj
                    
            elif occur[mi]<1:
                raise KeyError("Somewhere there must be an error. You chose a node which doesn't belong to any community!")
        elif mj!=m_empty:
            k_to_mi_mj = extract_neighbours(A_indices, A_indptr, k_to_mi_mj, l, c, mi, mj)
            if k_to_mi_mj[1]==0:
                continue
            
            # Metropolis
            if gamma is None: 
                deltaQ = delta_Q(K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)
            else:
                deltaQ = delta_Q_gamma(gamma, K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)    
            # if rn.random() < np.min(1, np.exp(beta*deltaQ)):
            if deltaQ<0: 
                    bad_deltaQ += deltaQ
                    cont += 1
            
            if occur[mi]>1:     # Condition (3)
                occur[mi]-=1
                occur[mj]+=1
                
                Kinn[mi]-=k[l]
                Kinn[mj]+=k[l]
                
                c[l]=mj
            elif occur[mi]==1:  # Condition (4)
                occur[mi]-=1
                occur[mj]+=1
                
                Kinn[mi]-=k[l]
                Kinn[mj]+=k[l]
                
                c[l]=mj
                shift(c=c, occur=occur, Kinn=Kinn, mi=mi, N=N)
                m_empty-=1

    if cont>0:
        bad_deltaQ = bad_deltaQ/cont
    else:
        raise ValueError("No se han contado cambios malos en el random walk, revísalo.")

    # Calculo beta0 con p0 (e.g. p0=0.85)
    beta0 = np.log(p0)/bad_deltaQ
    
    return beta0, bad_deltaQ

def SA_core(beta0, alpha, Q0, frozen_condition, N, c, m_empty, occur, A_indices, A_indptr, K_inv, K2K_inv, k, Kinn, k_to_mi_mj, rn, print_vals, betamax, constant_T_iters:int|None=None, max_cooling_steps:int|None=None, gamma:float|None=None):
    
        if constant_T_iters is None:
            iters: int = N
        else:
            iters: int = constant_T_iters
        
        beta=beta0
        Q=Q0
        cooling_steps=0
        frozen = False
        
        # The algorithm must register the community structure c which holds the maximum modularity seen in all iterations, Q_max. Therefore, we must define c_max, Q_max, which will be updated accordingly.
        c_max = np.zeros(N,np.int32)
        Q_max = -5.0
        
        # Initialize rejections. Whenever N consecutive rejections are reached, the algorithm stops.
        rejections = 0

        while not frozen:
            
            if betamax is not None:
                if beta>betamax: break

            # Loop of fixed temperature of N questions
            for _ in range(iters):
                # Question -----------------------------------------------------------------------------------------------------------------------------------------------
                
                # Choose randomly both a node that belongs to some community mi, as well as some mj!=mi
                l = rn.integers(0,N-1,endpoint=True)
                mi = c[l]
                mj = mi
                while mj==mi:
                    if m_empty<N: 
                        mj = rn.integers(0, m_empty, endpoint=True)
                    else:
                        mj = rn.integers(0, m_empty-1, endpoint=True)
                
                if mj==m_empty:         # If accepted, the chosen node l will be assigned the empty community mj
                    
                    if occur[mi]==1:    # Condition (1)
                        mj=mi

                    elif occur[mi]>1:   # Condition (2)
                        # Accept proposal with Metropolis transition probability
                        k_to_mi_mj = extract_neighbours(A_indices, A_indptr, k_to_mi_mj, l, c, mi, mj)
                        if gamma is None: 
                            deltaQ = delta_Q(K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)
                        else:
                            deltaQ = delta_Q_gamma(gamma, K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)
                        
                        if deltaQ >= 0 or rn.random() < np.exp(beta * deltaQ):  # Accepted
                            occur[mi]-=1
                            occur[mj]+=1
                            m_empty+=1
                            Kinn[mi]-=k[l]
                            Kinn[mj]+=k[l]
                            c[l]=mj
                            Q+=deltaQ
                            rejections=0
                        else:                                                   # Rejected
                            rejections+=1
                            if rejections==frozen_condition: 
                                frozen=True
                                break

                    elif occur[mi]<1:
                        raise KeyError("Somewhere there must be an error. You chose a node which doesn't belong to any community!")

                elif mj!=m_empty:
                    
                    # Here we must check whether there is at least one link between node l (of community mi) and any node of community mj (k_to_mi_mj[1]). If not, ignore the conditionals below and skip to next iteration. If there exist one link or more, proceed with Metropolis decission. This is because it would be very unlikely to increase modularity when transfering a node to a community with no connections.
                    k_to_mi_mj = extract_neighbours(A_indices, A_indptr, k_to_mi_mj, l, c, mi, mj)
                    if k_to_mi_mj[1]==0:
                        # We do count this possibility as a rejection
                        rejections+=1
                        continue
                    
                    # Accept proposal with Metropolis transition probability
                    if gamma is None: 
                        deltaQ = delta_Q(K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)
                    else:
                        deltaQ = delta_Q_gamma(gamma, K_inv, K2K_inv, k, l, mi, mj, k_to_mi_mj, Kinn)
                
                    if deltaQ >= 0 or rn.random() < np.exp(beta * deltaQ):      # Accepted
                        
                        if occur[mi]>1:     # Condition (3)
                            occur[mi]-=1
                            occur[mj]+=1
                            Kinn[mi]-=k[l]
                            Kinn[mj]+=k[l]
                            c[l]=mj

                        elif occur[mi]==1:  # Condition (4)
                            occur[mi]-=1
                            occur[mj]+=1
                            Kinn[mi]-=k[l]
                            Kinn[mj]+=k[l]
                            c[l]=mj
                            shift(c=c, occur=occur, Kinn=Kinn, mi=mi, N=N)
                            m_empty-=1

                        Q+=deltaQ
                        rejections=0
                    else:                                                       # Rejected
                        rejections+=1
                        if rejections==frozen_condition: 
                            frozen=True
                            break

                    if m_empty==N: 
                        raise ValueError("El algoritmo ha evolucionado hasta un punto en el que cada nodo está en una comunidad distinta. Algo va mal.")

                if Q>Q_max:
                    Q_max=Q
                    c_max[:]=c
                # End of question ----------------------------------------------------------------------------------------------------------------------------------------

            beta=alpha*beta
            cooling_steps+=1
            
            if max_cooling_steps is not None:
                if cooling_steps>=max_cooling_steps:
                    break
            
            if print_vals is not None:
                if print_vals==True:
                    if cooling_steps%100==0:
                        print(f"cooling_steps={cooling_steps}")
                        print(f"beta={beta}")
                        print(f"Q_max={Q_max}")
            
            if not np.isfinite(beta):
                frozen = True
                break
        
        if print_vals is not None and print_vals==True: print(f"Finish! cooling_steps={cooling_steps}")
        
        return c_max, Q_max

# Mother function
def simulated_annealing(Gf:nx.Graph, rn, T_random:int, p0:float, frozen_condition, alpha:float|None=None, S:int|None=None, pS:float|None=None, print_vals:bool|None=None, betamax:float|None=None, constant_T_iters:int|None=None, max_cooling_steps:int|None=None, gamma:float|None=None) -> tuple[np.ndarray,float]:
    r"""Apply simulated annealing for finding the optimal community structure of G.

    ## Args:
        Gf (nx.Graph): Disordered graph
        rn (_type_): random number generator
        T_random (int): number of random configurations that the will be swept in the first loop to find the initial temperature. Context: Initial beta (beta0) (inverse temperature) factor is estimated so that in the first sweep of N questions, 'bad' proposals are accepted with prob p0 (e.g. p0=0.85). To characterize what a bad proposal is, we must run a quick loop of T_random (e.g. 200) iterations in which we accept all configurations (of c) and register all values of deltaQ that are <0 (that is a undesireable proposal), and then we define the 'bad' proposal as the typical value of undesirable proposals deltaQ < 0.
        p0 (float): probability of accepting random ('bad') proposals in the loop for estimating beta0.
        alpha (float | None, optional): factor for changing beta in the cooling scheme: beta^(t) = beta^(t-1)*alpha. Defaults to None. A heuristic cooling factor alpha can be used, usually alpha \in [1.01, 1.05] is taken. The closer it is to 1, the better the results at the cost of more computation time. Note, that number can be motivated by arguing that after (e.g.) 200 cooling steps, the probability p_200 of accepting a bad configuration (one with deltaQ=bad_deltaQ) is 0.1%, i.e. p_200 = 0.001 (e.g.). The formula is in my iPad notes, page 11.
        S (int | None, optional): pasos de enfriamiento necesarios para que la probabilidad de aceptar configuraciones que empeoran 'mucho' la modularidad (en concreto, la empeoran deltaQ = bad_deltaQ) sea p_S. Defaults to None.
        pS (float | None, optional): see description of S. Defaults to None.
        
        ### Note
        With alpha=1.001 it works well. If you do not want to give a value for alpha, these values allow obtaining a very similar alpha: S=200; pS=0.001.

    ## Returns:
        tuple[np.ndarray,float]: (c_max, Q_max) correspond to the best community structure and modularity found in the process. c_max[i] gives the community label for node i.
    """
    n = np.array(list(Gf.nodes()))
    N = len(n)

    # Number of initial communities
    m = rn.integers(2, N-1, endpoint=True)  # Ex: m=2, then available indices are {0,1}, but initially 1 is assigned to m_empty, so c_i ~ U[0,0=m-2]

    # Initial assignation of community indices
    c = rn.integers(0, m-2, endpoint=True, size=N, dtype=np.int32)
    c_original = np.zeros(N,dtype=np.int32)

    # Canonize the sequence and count occurrences           # Ex:     [2 2 3 2 3 6 3 5 7 1 2 4]
    c2 = np.zeros(N, dtype=np.int32)                        #      -> [0 0 1 0 1 2 1 3 4 5 0 6] = L
    d = {}                                                  # occur = [4 3 1 1 1 1 1 0 0 0 0 0]

    ## occur[i] gives the number of occurrencies of community label i in c. Ex: in L, occur[0]=4, occur[6]=1. Its dimension is well-defined because the maximum number of communities is N (each node in a different community), so the maximum vector for storing N communities must be N-dim (occur[0], ..., occur[N-1]).
    occur = np.zeros(N, dtype=np.int32)
    occur_original = np.zeros(N, dtype=np.int32)
    sec=0
    for i, ci in enumerate(c):
        if d.get(ci) is None:
            d[ci]=sec
            sec+=1
        c2[i]=d[ci]
        occur[d[ci]]+=1
    c[:] = c2
    c_original[:] = c
    occur_original[:] = occur

    m_empty = max(c)+1
    m_empty_original = m_empty

    # Calculation of modularity for initial partition c
    assign_partition(G=Gf, c=c)
    comm = [set(n for n,d in Gf.nodes(data=True) if d['block']==i) for i in range(m_empty)]
    Q0 = modularity(Gf, communities=comm, resolution=gamma if gamma is not None else 1.0)

    # Calculation of the degree of nodes (k:array, k[i] gives the degree of node i) and number of edges (K:int)
    k = np.array([Gf.degree[ni] for ni in Gf.nodes()], dtype=np.int32)                                                       # IMP: NOT WEIGHTED GRAPHS ARE ASSUMED
    K = int(np.sum(k)*0.5)
    K_inv = 1./K
    K2K_inv = 1./(2*K*K)

    # Calculation of the inner degree of each initial community (Kinn:array, Kinn[mi]=sum of all degrees of nodes from community mi)
    Kinn = calculate_inner_degrees(c=c, k=k, N=N)
    Kinn_original = np.zeros(N, dtype=np.int32)
    Kinn_original[:] = Kinn
    ## After this initial calculation, Kinn will be updated in this manner (example): l=1 (chosen node), mi=c[l], mj=2 (random), Kinn[mj]+=k[l], Kinn[mi]-=k[l]

    # Initialization of k_to_mi_mj. k_to_mi_mj[0] gives the number of links from node l to community mi. k_to_mi_mj[1] gives the number of links from node l to community mj.
    k_to_mi_mj = np.zeros(2,np.int32)
    # Ex: k_to_mi_mj = extract_neighbours(A_indices, A_indptr, k_to_mi_mj, l, c, mi, mj)

    ## Now we need to know to which nodes is some node l connected. See details in the docstring of the function below.

    ## Convert to sparse CSR matrix in SciPy (only once per graph!)
    A_sparse = nx.to_scipy_sparse_array(Gf, format='csr')

    ## Extract the NumPy arrays forcing int32 for cache optimization
    A_indices = A_sparse.indices.astype(np.int32)
    A_indptr = A_sparse.indptr.astype(np.int32)
    
    # Here goes the random SA (everything is accepted) -------------------------------------------------------------------------------------------------------------------------
    
    # Initial beta (inverse temperature) factor is estimated so that in the first sweep of N questions, 'bad' proposals are accepted with prob p0 (e.g. p0=0.85). To characterize what a bad proposal is, we must run a quick loop of T_random (e.g. 200) iterations in which we accept all configurations (of c) and register all values of deltaQ that are <0 (that is a undesireable proposal), and then we define the 'bad' proposal as the typical value of undesirable proposals deltaQ < 0.

    ## Here you must make a function that runs the algorithm randomly (with a transfer probability = 1, accept everything recklessly), and that returns the average deltaQ.
    # T_random = 200  # Number of random configurations that the will be swept in the loop
    
    beta0, bad_deltaQ = SA_randwalk(T_random=T_random, p0=p0, N=N, c=c, m_empty=m_empty, occur=occur, A_indices=A_indices, A_indptr=A_indptr, K_inv=K_inv, K2K_inv=K2K_inv, k=k, Kinn=Kinn, k_to_mi_mj=k_to_mi_mj, rn=rn, gamma=gamma)
    
    # I use a heuristic cooling factor alpha, usually alpha \in [1.01, 1.05] is taken. The closer it is to 1, the better the results at the cost of more computation time.

    # Note, that number can be motivated by arguing that after (e.g.) 200 cooling steps, the probability p_200 of accepting a bad configuration (one with deltaQ=bad_deltaQ) is 0.1%, i.e. p_200 = 0.001 (e.g.). The formula is in my iPad notes, page 11.

    # S = 200 # cooling steps necessary so that the prob of accepting configurations that worsen modularity 'a lot' (specifically, they worsen it deltaQ = bad_deltaQ) is p_S
    # pS = 0.001
    
    if alpha is None:
        if S and pS is None:
            raise ValueError("S and pS are mandatory arguments if 'alpha' is not specified.")
        elif S and pS is not None:
            alpha = (np.log(pS)/(beta0*bad_deltaQ))**(1.0/S)
        else:
            raise ValueError("S or pS is missing!")
    
    # alpha = 1.001
    
    c[:] = c_original
    m_empty = m_empty_original
    occur[:] = occur_original
    Kinn[:] = Kinn_original
    
    if print_vals is not None:
        if print_vals==True:
            print(f"beta0={beta0}")
            print(f"alpha={alpha}")
    
    # SA -------------------------------------------------------------------------------------------------------------------------------------------------------------

    c_max, Q_max = SA_core(beta0, alpha, Q0, frozen_condition=frozen_condition, N=N, c=c, m_empty=m_empty, occur=occur, A_indices=A_indices, A_indptr=A_indptr, K_inv=K_inv, K2K_inv=K2K_inv, k=k, Kinn=Kinn, k_to_mi_mj=k_to_mi_mj, rn=rn, print_vals=print_vals, betamax=betamax, constant_T_iters=constant_T_iters, max_cooling_steps=max_cooling_steps, gamma=gamma)
    
    # End of SA ------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Canonize c_max (c_max so that c[0]=0 always, and c[-1]=max(c), in order)
    sec=0
    d={}
    for i, ci in enumerate(c_max):
        if d.get(ci) is None:
            d[ci]=sec
            sec+=1
        c_max[i]=d[ci]
    
    return c_max, Q_max

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

def numba_warm_up(rn):
    print("Compiling Numba...")
    Gi, Qi = sbm_gn([16,16], k=5, k_out=1)
    Gf, _ = shuffled_graph(Gi, rn=rn, shuffle=True)
    c_max, Q_max = simulated_annealing(Gf, rn, T_random=100, p0=0.85, alpha=1.001, frozen_condition=3)
    print("Done!\n")