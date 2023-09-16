import warnings
warnings.filterwarnings('ignore')
import sys
# sys.path.append("../")
import osmnx as ox
# import torch as th
# import networkx as nx
# import dgl
# import matplotlib.pyplot as plt


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    import sys

    # sys.path.append("../")
    import osmnx as ox
    # import torch as th
    # import networkx as nx
    # import dgl
    # import matplotlib.pyplot as plt

    streets_graph = ox.graph_from_place('Los Angeles, California', network_type='drive')#address：Los Angeles城市名，California
    new_york_path = ox.graph_from_place('New York City, New York, USA', network_type='drive', simplify=True)
    G = ox.plot_graph(new_york_path)
    ox.io.save_graphml(new_york_path, filepath='new_york_street_gephi_false_wi_simplify.graphml', gephi=False,
                       encoding='utf-8')
