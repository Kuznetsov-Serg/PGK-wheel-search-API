import networkx as nx
import matplotlib.pyplot as plt

# Graph data
names = ['A', 'B', 'C', 'D', 'E']
positions = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1, 1)]
edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('D', 'A')]

# Matplotlib figure
plt.figure('My graph problem')

# Create graph
G = nx.MultiDiGraph(format='png', directed=True)

for index, name in enumerate(names):
    G.add_node(name, pos=positions[index])

labels = {}




layout = dict((n, G._node[n]["pos"]) for n in G.nodes())
nx.draw(G, pos=layout, with_labels=True, node_size=300)
ax = plt.gca()
for edge in edges:
    ax.annotate("",
                xy=layout[edge[0]], xycoords='data',
                xytext=layout[edge[1]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=-0.3",
                                ),
                )
plt.show()