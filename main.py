# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
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
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
