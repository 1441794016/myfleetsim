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
    manhattan_path = ox.graph_from_place('Manhattan, New York City, New York, USA', network_type='drive')
    G = ox.plot_graph(manhattan_path)

