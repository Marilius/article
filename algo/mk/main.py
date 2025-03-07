from algo.helpers import input_graph, calc_cut_ratio, do_unpack_mk, unpack_mk

import algo.base_partitioner.settings as settings

from algo.greed.main import GreedPartitioner

import networkx as nx

from os import makedirs
from os.path import isfile, join

import time


class MKPartitioner(GreedPartitioner):
    def write_mk(self, g_name: str, G_grouped: nx.Graph, mk_partition: list[int], cr_max: float) -> None:
        """
        Writes a coarsened graph to a file and information needed for further uncoarsening.

        Args:
            g_name (str): The name of the graph file.
            G_grouped (nx.Graph): The coarsened graph.
            mk_partition (list[int]): The coarsening information.
            cr_max (float): The maximum allowed cut ratio.

        Returns:
            None
        """
        if cr_max == 1:
            cr_max = 1

        output_file = settings.MK_DIR + g_name.replace('.', str(cr_max) + '.')

        with open(output_file, 'w+') as file:
            file.write('name weight children\n')

            for node_id in sorted(list(G_grouped.nodes)):
                line = [str(node_id), str(G_grouped.nodes[node_id]['weight'])]

                for neighbor in G_grouped.neighbors(node_id):
                    if neighbor > node_id:
                        line.append(str(neighbor))

                line.append('\n')
                file.write(' '.join(line))

        ending2replace = '.txt' if 'txt' in g_name else '.graph'
        output_file = settings.MK_DIR + g_name.replace(ending2replace, str(cr_max) + '.' + 'mapping')
        with open(output_file, 'w+') as file:
            file.write(' '.join(map(str, mk_partition)))

    def load_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool, steps_back: int) -> list[int] | None:
        """
        Loads result of mk_nparts function from the cache for a given graph, the number of parts, the cut ratio, whether the graph is weighted and the number of steps back.

        Args:
            G (nx.Graph): The graph.
            nparts (int): The number of parts.
            cr (float): The cut ratio.
            weighted (bool): Whether the graph is weighted.
            steps_back (int): The number of steps back.

        Returns:
            list[int] | None: The result of mk_nparts function call if it exists in the cache, otherwise None.
        """

        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = f'{settings.CACHE_DIR}/{G.graph["graph_name"]}_!{str(nparts)}!_{w}{str(steps_back)}!_{str(cr)}_{str(weighted)}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition

        return None

    def write_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool, partition: list[int] | None, steps_back: int) -> None:        
        """
        Writes result of mk_nparts function to the cache for a given graph, the number of parts, the cut ratio, whether the graph is weighted and the number of steps back.

        Args:
            G (nx.Graph): The graph.
            nparts (int): The number of parts.
            cr (float): The cut ratio.
            weighted (bool): Whether the graph is weighted.
            partition (list[int] | None): The result of mk_nparts function call.
            steps_back (int): The number of steps back.

        Returns:
            None
        """
        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = f'{settings.CACHE_DIR}/{G.graph["graph_name"]}_!{str(nparts)}!_{w}{str(steps_back)}!_{str(cr)}_{str(weighted)}.txt'
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def mk_nparts(
        self,
        G: nx.Graph,
        nparts: int,
        cr_max: float,
        check_cache: bool,
        seed: int | None,
        max_ufactor: float | None = 1e7,
        weighted: bool = True,
        steps_back: int = 5
    ) -> tuple[nx.Graph, list[int]] | tuple[None, None]:
        """
        Runs the MK partitioning algorithm on a given graph with a given number of parts and maximum allowed cut ratio.

        Args:
            G (nx.Graph): The graph to be coarsened.
            nparts (int): The number of nodes in coarsened graph.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check the cache before running the algorithm.
            seed (int | None): The seed for the algorithm.
            max_ufactor (float | None): The maximum allowed ufactor for METIS.
            weighted (bool): Whether the graph is weighted.
            steps_back (int): The number of steps back.

        Returns:
            tuple[nx.Graph, list[int]] | tuple[None, None]: A tuple containing the coarsened graph and the partition assignment if the cut ratio is satisfied, otherwise None.
        """
        if max_ufactor is not None:
            self.MAX_UFACTOR = max_ufactor
        if check_cache:
            partition = self.load_mk_nparts_cache(G, nparts, cr_max, weighted, steps_back=steps_back)
            if partition is not None:
                G_grouped = self.group_mk(G, partition, weighted=weighted)
                G_grouped.graph['graph_name'] = f'{G.graph["graph_name"]}_grouped_{str(nparts)}_{str(cr_max)}_{str(weighted)}'
                return (G_grouped, partition)
        partition_ans = super().do_metis(G, nparts, cr_max, check_cache, seed, steps_back=steps_back)

        if partition_ans is None:
            return (None, None)

        G_grouped = self.group_mk(G, partition_ans, weighted=weighted)

        print('--->', G_grouped.nodes)
        print('--->', partition_ans)

        if check_cache:
            self.write_mk_nparts_cache(G, nparts, cr_max, weighted, partition_ans, steps_back=steps_back)

        G_grouped.graph['graph_name'] = f'{G.graph["graph_name"]}_grouped_{str(nparts)}_{str(cr_max)}_{str(weighted)}'

        return (G_grouped, partition_ans)

    def load_mk_part_cache(self, G: nx.Graph, cr_max: float, steps_back: int) -> list[int] | None:
        """
        Loads result of mk_part function from the cache for a given graph, the maximum allowed cut ratio and the number of steps back.

        Args:
            G (nx.Graph): The graph.
            cr_max (float): The maximum allowed cut ratio.
            steps_back (int): The number of steps back.

        Returns:
            list[int] | None: The result of mk_part function call if it exists in the cache, otherwise None.
        """
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/mk_part/{G_hash}_{cr_max}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                partition = list(map(int, line.split()))
                return partition

        return None

    def write_mk_part_cache(self, G: nx.Graph, partition: list[int], cr_max: float, steps_back: int) -> None:
        """
        Writes the result of the mk_part function call to the cache for a given graph,
        partition, maximum allowed cut ratio, and number of steps back.

        Args:
            G (nx.Graph): The input graph.
            partition (list[int]): The partitioning result where each node is assigned
                to a partition.
            cr_max (float): The maximum allowed cut ratio.
            steps_back (int): The number of steps back.

        Returns:
            None
        """

        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/mk_part/{G_hash}_{cr_max}_{steps_back}.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            file.write(' '.join(map(str, partition)))

    def mk_part(self, G: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 5) -> tuple[int, list[int]]:
        """
        Divides a graph into the maximum number of graph parts that satisfy the constraint on the cut ratio.

        Args:
            G (nx.Graph): The graph to be partitioned.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check the cache before calling METIS.
            seed (int | None): The seed for METIS.
            steps_back (int): The number of times of ufactor reducing.

        Returns:
            tuple[int, list[int]]: A tuple containing the number of parts and a list of the partition assignment for each node if the cut ratio is satisfied, otherwise None.
        """
        if check_cache:
            partition = self.load_mk_part_cache(G, cr_max, steps_back=steps_back)
            if partition is not None:
                return (len(set(partition)), partition)

        num_left = 1
        num_right = len(G)

        n_ans = 0
        partition_ans = None

        n = 0
        ufactor = 0

        while num_right - num_left > 0:
            n = (num_left + num_right) // 2
            print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, settings.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

            ufactor = 1
            while True: 
                (_, partition) = self.metis_part(G, n, ufactor, check_cache, seed)
                print(n, len(set(partition)), ufactor, calc_cut_ratio(G, partition))
                if self.check_cut_ratio(G, partition, cr_max):
                    print('was here')
                    partition_curr = partition.copy()

                    for _ in range(steps_back):
                        ufactor *= 0.75
                        ufactor = int(ufactor)
                        if ufactor < 1:
                            break

                        (_, partition) = self.metis_part(G, n, ufactor, check_cache, seed)
                        if len(set(partition_curr)) < len(set(partition)):
                            partition_curr = partition.copy()

                    if len(set(partition_curr)) > n_ans:
                        num_left = len(set(partition_curr)) + 1
                        n_ans = len(set(partition_curr))
                        partition_ans = partition_curr
                    else:
                        num_right = n

                    break

                if ufactor > settings.MAX_UFACTOR:
                    print('ENDED BY UFACTOR')
                    num_right = n - 1
                    break

                ufactor += ufactor

        print('main ended')

        ufactor = 1
        while ufactor < settings.MAX_UFACTOR:
            (_, partition) = self.metis_part(G, num_right, ufactor, check_cache, seed)
            if self.check_cut_ratio(G, partition, cr_max):
                if len(set(partition)) > n_ans:
                    n_ans = len(set(partition))
                    partition_ans = partition
                break
            ufactor *= 2

        if set(range(n_ans)) != set(partition_ans):
            mapping = dict()

            for new_id, old_id in enumerate(set(partition_ans)):
                mapping[old_id] = new_id

            for i in range(len(partition_ans)):
                partition_ans[i] = mapping[partition_ans[i]]

        if check_cache:
            self.write_mk_part_cache(G, partition_ans, cr_max, steps_back=steps_back)

        return (n_ans, partition_ans)

    def get_num_mk(self, G: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 5, ) -> int:
        """
        Returns the maximum number of parts into which a graph can be divided while still satisfying the constraint

        Args:
            G (nx.Graph): The graph to be partitioned.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check the cache before calling METIS.
            seed (int | None): The seed for METIS.
            steps_back (int): The number of times of ufactor reducing.

        Returns:
            int: the maximum number of parts into which a graph can be divided.
        """
        (n, _) = self.mk_part(G, cr_max, check_cache, seed, steps_back=steps_back)

        return n

    def group_mk(self, G: nx.Graph, partition: list[int], weighted: bool = True) -> nx.Graph:
        """
        Groups nodes in a graph based on a given partition, creating a new coarsened graph.

        Args:
            G (nx.Graph): The original graph to be grouped.
            partition (list[int]): A list where each index corresponds to a node in the graph,
                and the value is the partition group the node belongs to.
            weighted (bool): Indicates whether to mark the grouped graph as weighted.

        Returns:
            nx.Graph: A new graph where nodes are grouped according to the partition, with
            edges between groups if there were edges between any nodes in the original graph.
        """

        grouped_G = nx.Graph()
        nodes_ids = sorted(list(set(partition)))

        for node_id in nodes_ids:
            weight = 0
            for num, part in enumerate(partition):
                if part == node_id:
                    weight += G.nodes[num]['weight']
            grouped_G.add_node(node_id, weight=weight)

        for old_node_id, node_id in enumerate(partition):
            for old_neighbor_id in G.neighbors(old_node_id):
                neighbor_id = partition[old_neighbor_id]
                if node_id != neighbor_id:
                    grouped_G.add_edge(node_id, neighbor_id)
            
        if weighted:
            grouped_G.graph['node_weight_attr'] = 'weight'

        grouped_G.graph['graph_name'] = G.graph['graph_name'] + '_grouped'

        return grouped_G


    def MK_greed_greed(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 6, ) -> list[int] | None:
        """
        Determines the maximum number of parts into which a graph can be divided so that the constraint on cut ratio is respected.
        
        then for each i from 1 to this number:
            The graph is partitioned into i parts without cut ratio restrictions and its vertices are grouped then assigns groups of tasks to processors according to their performance.
            Coarsened graph partition is refined by greed algorithm without cut ratio restrictions.
            Coarsened graph and it's partition is being unpacked and further refined by greed algorithm.

        The best graph partition found is being returned.        

        Args:
            G (nx.Graph): The graph to be partitioned.
            PG (nx.Graph): The physical graph.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check the cache before running the algorithm.
            seed (int | None): The seed for the algorithm.
            steps_back (int): The number of steps back.

        Returns:
            list[int] | None: The best partition found, or None if no valid partition was found.
        """
        max_mk = self.get_num_mk(G, cr_max, check_cache, seed, steps_back=steps_back)

        best_partition: list[int] = [0] * len(G.nodes)
        best_f: float = self.f(G, PG, best_partition, cr_max)

        n = 0
        if self.do_metis_with_pg(G, PG, cr_max, check_cache, seed):
            part = self.do_metis_with_pg(G, PG, cr_max, check_cache, seed)
            if part is not None:
                n = len(set(part))
            else:
                n = 1

        if n <= max_mk:
            print('WARNING: n < MK_MAX ', n, max_mk, G.graph['graph_name'], cr_max)

        for nparts in range(1, max_mk + 1):
            (G_grouped, mk_data) = self.mk_nparts(G, nparts, cr_max, check_cache, seed, steps_back=steps_back)

            if G_grouped is None or mk_data is None:
                continue

            mk_partition = self.do_metis_with_pg(G_grouped, PG, 1, check_cache, seed, steps_back=steps_back)
            mk_partition = self.do_greed(G_grouped, PG, mk_partition, 1)

            if mk_partition is None:
                continue
            
            try:
                mk_partition_unpacked = unpack_mk(mk_partition, mk_data)
            except Exception as e:
                print(mk_partition, mk_data)
                raise e
            if self.check_cut_ratio(G, mk_partition_unpacked, cr_max):
                partition = self.do_greed(G, PG, mk_partition_unpacked, cr_max)
                f_val = self.f(G, PG, partition, cr_max)

                if f_val < best_f:
                    best_f = f_val
                    best_partition = partition.copy()

        return best_partition
    
    def do_MK_greed_greed(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
        steps_back: int,
        check_cache: bool,
        seed: int | None,
    ) -> list[int] | None:
        """
        Run the MK_greed_greed algorithm on a weighted graph and write the results to a file.

        Args:
            input_dir (str): The directory containing the input graph.
            output_dir (str): The directory where the partitioning results will be written.
            graph_file (str): The name of the input graph file.
            physical_graph_dir (str): The directory containing the physical graph.
            physical_graph_path (str): The path to the physical graph file.
            cr_max (float): The maximum allowed cut ratio.
            steps_back (int): The number of steps to go back in the MK algorithm.
            check_cache (bool): Whether to check if the result is already in the cache.
            seed (int | None): The seed to use for the MK algorithm.

        Returns:
            list[int] | None: The partitioning result or None if the cut ratio is not satisfied.
        """

        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'MK_greed_greed_weighted')

        start_time = time.time()
        partition = self.MK_greed_greed(weighted_graph, physical_graph, cr_max, check_cache, seed, steps_back=steps_back)

        # assert self.f(weighted_graph, physical_graph, partition) <= self.f(weighted_graph, physical_graph, self.just_weighted_partition)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, cr_max, start_time)

        return partition

    def MK_greed_greed_with_geq_cr(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, steps_back: int = 6) -> list[int]:
        best_best_partition: list[int] = [0] * len(G.nodes)
        best_best_f: float = self.f(G, PG, best_best_partition, cr_max)

        max_mk = self.get_num_mk(G, cr_max, check_cache, seed, steps_back=steps_back)

        n = 0
        if self.do_metis_with_pg(G, PG, cr_max):
            n = len(set(self.do_metis_with_pg(G, PG, cr_max)))
        
        if n <= max_mk:
            print('WARNING: n < MK_MAX ', n, max_mk, G.graph['graph_name'], cr_max)

        for cr in settings.ALL_CR_LIST:
            if cr >= cr_max:
                best_partition: list[int] = [0] * len(G.nodes)
                best_f: float = self.f(G, PG, best_partition, cr)

                f: bool = False

                for nparts in range(1, max_mk + 1):
                    (G_grouped, mk_data) = self.mk_nparts(G, nparts, cr, check_cache=check_cache, steps_back=steps_back)

                    if G_grouped is None or mk_data is None:
                        continue

                    mk_partition = self.do_metis_with_pg(G_grouped, PG, 1, check_cache=check_cache, steps_back=steps_back)
                    mk_partition = self.do_greed(G_grouped, PG, mk_partition, 1)

                    if mk_partition is None:    
                        continue

                    mk_partition_unpacked = unpack_mk(mk_partition, mk_data)
                    if not self.check_cut_ratio(G, mk_partition_unpacked, cr_max):
                        f = True
                        break

                    if self.check_cut_ratio(G, mk_partition_unpacked, cr_max):
                        partition = self.do_greed(G, PG, mk_partition_unpacked, cr_max)
                        f_val = self.f(G, PG, partition, cr_max)
                        if f_val < best_f:
                            best_f = f_val
                            best_partition = partition.copy()

                if best_f < best_best_f:
                    best_best_f = best_f
                    best_best_partition = best_partition.copy()

                if f:
                    break

        if best_f < best_best_f:
            best_best_f = best_f
            best_best_partition = best_partition.copy() 

        return best_best_partition

    def do_MK_greed_greed_with_geq_cr(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
        seed: int | None,
        check_cache: bool,
        steps_back: int = 6,
    ) -> None:
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'MK_greed_greed_with_geq_cr')

        start_time = time.time()
        partition = self.MK_greed_greed_with_geq_cr(weighted_graph, physical_graph, cr_max, seed, check_cache=check_cache, steps_back=steps_back)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, cr_max, start_time)

    def do_weighted_mk_with_geq_cr(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
    ) -> None:
        output_dir = output_dir.replace('results', 'results1')
        weighted_graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'greed_from_mk_weighted')

        ans_init = None
        ans_part = None
        ans = None

        for cr in settings.ALL_CR_LIST:
            if cr >= cr_max:
                mk_path = settings.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                mk_data_path = settings.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')

                if not isfile(mk_path) or not isfile(mk_data_path):
                    continue

                mk_graph_weighted = input_graph(mk_path)

                initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, 1, check_cache=True)
                assert initial_weighted_partition is not None
                weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition, 1)
                assert weighted_partition is not None

                initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                assert initial_weighted_partition is not None
                weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                assert weighted_partition is not None

                if not self.check_cut_ratio(weighted_graph, weighted_partition, cr_max):
                    break

                if ans_part is None or ans is None or self.f(weighted_graph, physical_graph, weighted_partition, cr_max) < ans:
                    ans_init = initial_weighted_partition
                    ans_part = weighted_partition
                    ans = self.f(weighted_graph, physical_graph, weighted_partition, cr_max)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), ans_init, weighted_graph, physical_graph, cr_max)
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), ans_part, weighted_graph, physical_graph, cr_max)
