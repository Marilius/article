import networkx as nx

from os.path import isfile

from time import sleep


def input_graph(path: str) -> nx.Graph:
    """
    Reads a graph from a file in either the format used by the task_graph_generator.py script or the
    "node_id node_weight child1 child2 ..." format. The graph is read into a NetworkX graph object.

    Args:
        path (str): path to the file containing the graph

    Returns:
        nx.Graph: the read graph
    """
    if 'gen_data' in path:
        graph, _, _ = input_generated_graph_and_processors_from_file(path)
    else:
        graph = input_networkx_graph_from_file(path)
        
    return graph

def input_networkx_graph_from_file(path: str) -> nx.Graph:
    """
    Reads a graph from a file in the "node_id node_weight child1 child2 ..." format. The graph is read into a NetworkX graph object.

    Args:
        path (str): path to the file containing the graph

    Returns:
        nx.Graph: the read graph
    """
    G = nx.Graph()

    if not isfile(path):
        if 'data_mk' in path:
            while not isfile(path):
                print('WAITING FOR MK GRAPH: ', path)
                sleep(10)

        raise FileNotFoundError(f'File {path} not found')

    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, size, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=size)
            G.add_edges_from((name, child) for child in children)
    G.graph['node_weight_attr'] = 'weight'
    
    mk = 'mk_' if 'data_mk' in path else ''

    graph_name = mk + path.split('/')[-1].split('.')[0]
    G.graph['graph_name'] = graph_name

    return G

def input_networkx_unweighted_graph_from_file(path: str) -> nx.Graph:
    """
    Reads an unweighted graph from a file in format
    "node_id node_weight child1 child2 ...". 
    The graph is read into a NetworkX graph object, and the node weights are set to 1.

    Args:
        path (str): path to the file containing the graph

    Returns:
        nx.Graph: the read graph
    """
    G = nx.Graph()

    if not isfile(path):
        if 'data_mk' in path:
            while not isfile(path):
                print('WAITING FOR MK GRAPH: ', path)
                sleep(10)

        raise FileNotFoundError(f'File {path} not found')

    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, _, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=1)
            G.add_edges_from((name, child) for child in children)

    graph_name = path.split('/')[-1].split('.')[0]
    mk = 'mk_' if 'data_mk' in path else ''

    G.graph['graph_name'] = mk + graph_name
    
    return G

def input_generated_graph_partition(path: str) -> list[int]:
    """
    Reads a partition from a file created by task_graph_generator.py.

    Args:
        path (str): The path to the file containing the partition.

    Returns:
        list[int]: The partition read from the file.

    Raises:
        FileNotFoundError: If the file specified by path does not exist.
    """
    if not isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    exact_partition: list[int] = []

    with open(path, 'r') as f:
        exact_partition = list(map(int, f.readline().strip().split()))

    return exact_partition


def input_generated_graph_and_processors_from_file(path: str) -> tuple[nx.Graph, list[int], dict[str, int|list[int]]]:
    """
    Reads a graph and processor details from a file created by task_graph_generator.py.

    Args:
        path (str): The path to the file containing the graph and processor information.

    Returns:
        tuple[nx.Graph, list[int], dict[str, int|list[int]]]: A tuple containing the graph as a 
        NetworkX object, a list of processors, and a dictionary of parameters including 'p', 'L', 
        'min_l', 'max_l', 'N_e', and 'N_s'.

    Raises:
        FileNotFoundError: If the file specified by path does not exist.
    """

    if not isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    G = nx.Graph()
    params: dict[str, int|list[int]] = {}

    with open(path, 'r') as f:
        p = list(map(int, f.readline().strip().split()))
        L = int(f.readline())
        min_l, max_l = list(map(int, f.readline().strip().split()))
        N_e, N_s = list(map(int, f.readline().strip().split()))

        params = {'p': p, 'L': L, 'min_l': min_l, 'max_l': max_l, 'N_e': N_e, 'N_s': N_s}

        for line in f.readlines():
            name, size, *children = map(int, line.strip().split())
            # name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=size)
            G.add_edges_from((name, child) for child in children)
    G.graph['node_weight_attr'] = 'weight'
    
    graph_name = path.split('/')[-1].split('.')[0]
    G.graph['graph_name'] = graph_name

    return G, p, params

def calc_edgecut(G: nx.Graph, partition: list[int]) -> int:
    """
    Calculate the number of edges crossing between parts in a graph partition.

    Args:
        G (nx.Graph): The graph to calculate the edgecut for.
        partition (list[int]): The partition of the graph.

    Returns:
        int: The number of edges crossing between different parts.
    """
    edgecut: int = 0
    for edge in G.edges:
        node1, node2 = edge
        if partition[node1] != partition[node2]:
            edgecut += 1

    return edgecut

def calc_cut_ratio(G: nx.Graph | None, partition: list[int] | None) -> float | None:
    """
    Calculate the cut ratio of a graph given a partitioning of the graph.

    The cut ratio is the number of edges crossing between different partitions
    divided by the total number of edges in the graph.

    Args:
        G (nx.Graph): The graph to calculate the cut ratio for.
        partition (list[int]): The partitioning of the graph.

    Returns:
        float | None: The cut ratio of the graph given the partitioning. If the
            graph or partition is None, returns None.
    """

    if G is None or partition is None:
        return None

    if len(G.edges) == 0:
        return 0

    return calc_edgecut(G, partition) / len(G.edges)

def unpack_mk(mk_partition: list[int], mk_data: list[int]) -> list[int]:
    """
    Unpacks a coarsened partition into its original form using mapping data.

    Args:
        mk_partition (list[int]): A list representing the coarsened partition,
            where each index corresponds to a coarsened node and the value is
            the processor assigned to that node.
        mk_data (list[int]): A list representing the original nodes' mapping
            to coarsened nodes.

    Returns:
        list[int]: A list representing the original partition where each index
            corresponds to an original node and the value is the processor
            assigned to that node.
    """

    ans = mk_data.copy()
    mapping: dict[int, int] = dict()

    for mk_id, proc in enumerate(mk_partition):
        mapping[mk_id] = proc

    for i in range(len(mk_data)):
        ans[i] = mapping[ans[i]]

    return ans

def do_unpack_mk(mk_partition: list[int], mk_data_path: str) -> list[int]:
    """
    Unpacks a coarsened partition into its original form using mapping data.

    Args:
        mk_partition (list[int]): A list representing the coarsened partition,
            where each index corresponds to a coarsened node and the value is
            the processor assigned to that node.
        mk_data_path (str): The path to the mapping data file.

    Returns:
        list[int]: A list representing the original partition where each index
            corresponds to an original node and the value is the processor
            assigned to that node.
    """
    while not isfile(mk_data_path):
        print('waiting for: ', mk_data_path)
        sleep(10)

    with open(mk_data_path, 'r') as file:
        line = file.readline()
        mk_data = list(map(int, line.split()))

        return unpack_mk(mk_partition, mk_data)

def fix_rand_graph_file(path: str) -> None:
    """
    Fixes the node identifiers in a graph file by ensuring they are contiguous
    and starting from 0. Reads the graph from the specified path, renames the
    node identifiers in the file, and writes the corrected graph back to the
    file.

    Args:
        path (str): The file path to the graph file that needs to be fixed.

    Raises:
        AssertionError: If the graph nodes are not contiguous and starting from 0
            after the fix.
    """

    graph: nx.Graph = input_graph(path)
    
    with open(path, 'r') as file:
        filedata = file.read()
        for old_id, new_id in zip(list(sorted(list(graph.nodes))), list(range(len(graph.nodes)))):
            filedata = filedata.replace(f'\t{old_id}\n', f'\t{new_id}\n')
            filedata = filedata.replace(f'\t{old_id}\t', f'\t{new_id}\t')
            filedata = filedata.replace(f'\n{old_id}\t', f'\n{new_id}\t')
            filedata = filedata.replace(f'\n{old_id}\n', f'\n{new_id}\n')

    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)
    
    graph: nx.Graph = input_graph(path)
    assert list(sorted(list(graph.nodes))) == list(range(len(graph.nodes)))
