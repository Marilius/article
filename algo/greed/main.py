from algo.base_partitioner.main import BasePartitioner

from algo.helpers import input_graph, input_networkx_unweighted_graph_from_file, calc_cut_ratio


from os import makedirs
from os.path import join

import networkx as nx

import time


class GreedPartitioner(BasePartitioner):
    def write_results(self, path: str, physical_graph_path: str, partition: list[int], G: nx.Graph, PG: nx.Graph, cr_max: float, start_time: float) -> None:
        """
        Writes partitioning results and timing information to specified files.

        This function records the results of a graph partitioning operation,
        including details about the graph, physical graph, cut ratio, function value,
        partition list, and execution time. The information is appended to the specified
        output files.

        Args:
            path (str): The file path to write the partitioning results.
            physical_graph_path (str): The path to the physical graph file.
            partition (list[int]): The partitioning result where each node is assigned
                to a partition.
            G (nx.Graph): The input graph.
            PG (nx.Graph): The physical graph.
            cr_max (float): The maximum allowed cut ratio.
            start_time (float): The start time of the partitioning operation.

        Raises:
            AssertionError: If the partition is not valid for the given graphs.
        """
        # Record the end time of the operation
        end_time = time.time()

        # Prepare the line to write to the results file
        line2write = [
            path.split('/')[-1],  # The graph name
            physical_graph_path.split('/')[-1],  # The physical graph name
            calc_cut_ratio(G=G, partition=partition),  # The cut ratio
            cr_max,  # The maximum allowed cut ratio
            self.f(G, PG, partition, cr_max),  # The function value
            partition if self.check_cut_ratio(G, partition, cr_max) else None,  # The partition list
            '\n',  # A newline at the end of the line
        ]

        # Assert that the partition is valid for the given graphs
        assert partition is None or len(set(partition)) <= len(PG.nodes)
        assert partition is None or len(partition) == len(G.nodes)

        # Make the directory if it doesn't exist and open the file for writing
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        with open(path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))

        # Calculate the overall time
        overall_time = end_time - start_time

        # Prepare the line to write to the timing file
        line2write = [
            path.split('/')[-1],  # The graph name
            physical_graph_path.split('/')[-1],  # The physical graph name
            cr_max,  # The maximum allowed cut ratio
            len(partition) if partition is not None else None,  # The number of partitions
            start_time,  # The start time
            str(overall_time),  # The overall time
            '\n',  # A newline at the end of the line
        ]

        # Make the directory if it doesn't exist and open the file for writing
        time_path = path.replace('.txt', '.time').replace('.graph', '.time')
        with open(time_path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))

    def postprocessing_phase(self, G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None, cr_max: float) -> list[int] | None:
        """
        This function implements the postprocessing phase of the graph partitioning algorithm.
        It aims to optimize the partition by reassigning tasks to processors in order to
        reduce the maximum load on any processor while respecting the cut ratio constraint.
        
        The postprocessing phase has the following scheme.
        1) Select the most loaded processor; denote it as P1;
        2) For each task A assigned to P1, in decreasing order by task execution time:
            a) Choose the fastest processor P2 of the processors meeting the following
                constraint: if the task A is reassigned from P1 to P2, then
                max(load of P1, load of P2) decreases, and the CR constraint is met;
            b) If such processor P2 was found, reassign A from P1 to P2; go to step 1;
                else stop considering tasks on P1 with the same execution time as A until
                return to step 1;
            c) if last of execution times for tasks on P1 was discarded in step b, then stop.

        Args:
            G (nx.Graph | None): The input graph with task weights.
            PG (nx.Graph): The physical graph with processor weights.
            partition (list[int] | None): The partition to refine.
            cr_max (float): The maximum allowed cut ratio.

        Returns:
            list[int] | None: The refined partition.
        """
        if partition is None or G is None:
            return None

        # Initialize processor loads and order by processor weight
        p_loads = [0] * len(PG)
        p_order: list[int] = list(range(len(PG)))
        p_order.sort(key=lambda i: PG.nodes[i]['weight'], reverse=True)

        # Calculate initial processor loads based on current partition
        for i in range(len(partition)):
            p_loads[partition[i]] += G.nodes[i]['weight']

        flag = True
        while flag:
            # Select the most loaded processor (P1)
            p1 = None
            p1_time = 0
            for i, i_load in enumerate(p_loads):
                if i_load / PG.nodes[i]['weight'] > p1_time or p1 is None:
                    p1 = i
                    p1_time = i_load / PG.nodes[i]['weight']

            if p1 is None:
                break

            while flag:
                flag = False

                # Find the task with the maximum weight assigned to P1
                a = None
                a_weight = 0
                for job, proc in enumerate(partition):
                    if proc == p1:
                        if G.nodes[job]['weight'] > a_weight:
                            a = job
                            a_weight = G.nodes[job]['weight']

                if a is None:
                    break

                # Attempt to reassign task A to another processor P2
                for proc in p_order:
                    if proc != p1:
                        # Check if reassigning improves load balance and meets CR constraint
                        if max(p_loads[proc] / PG.nodes[proc]['weight'], p_loads[p1] / PG.nodes[p1]['weight']) > \
                                max((p_loads[p1] - a_weight) / PG.nodes[p1]['weight'], (p_loads[proc] + a_weight) / PG.nodes[proc]['weight']):
                            partition_copy = partition.copy()
                            partition_copy[a] = proc
                            if self.check_cut_ratio(G, partition_copy, cr_max):
                                # Update processor loads and partition
                                p_loads[proc] += a_weight
                                p_loads[p1] -= a_weight
                                partition[a] = proc
                                flag = True
                                break

                if flag:
                    break

        return partition

    def do_greed(self, G: nx.Graph, PG: nx.Graph, partition: list[int] | None, cr_max: float) -> list[int] | None:
        """
        Run the greedy partitioning algorithm on the given graph and physical graph.

        Args:
            G (nx.Graph): The graph to be partitioned.
            PG (nx.Graph): The physical graph.
            partition (list[int] | None): An initial partition assignment for each node.
            cr_max (float): The maximum allowed cut ratio.

        Returns:
            list[int] | None: The partition assignment for each node if the cut ratio is satisfied, otherwise None.
        """
        if partition is None:
            return None

        print('BASE', 'cr:', calc_cut_ratio(G, partition))
        weights = [0] * len(PG)
        for i in range(len(partition)):
            weights[partition[i]] += G.nodes[i]['weight']
        print('BASE', weights)
        print(self.f(G, PG, partition, cr_max))

        partition = self.postprocessing_phase(G, PG, partition, cr_max)
        print('GREED', 'cr:', calc_cut_ratio(G, partition))
        weights = [0] * len(PG)
        for i in range(len(partition)):
            weights[partition[i]] += G.nodes[i]['weight']
        print('GREED', weights)
        print(self.f(G, PG, partition, cr_max))

        return partition

    def simple_part(self, G: nx.Graph, PG: nx.Graph) -> list[int]:
        """
        Assigns all tasks in the input graph to the fastest processor.

        Args:
            G (nx.Graph): The input graph where each node represents a task.
            PG (nx.Graph): The physical graph where each node represents a processor with a weight.

        Returns:
            list[int]: A partition list where each task in the input graph is assigned to the
            fastest processor.
        """

        proc_fastest: int = 0
        speed_max: int = PG.nodes[proc_fastest]['weight']

        for proc in PG.nodes:
            speed = PG.nodes[proc]['weight']
            if speed > speed_max:
                proc_fastest = proc
                speed_max = speed

        return [proc_fastest] * len(G)

    def do_simple_part(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
    ) -> None:
        """
        Assigns all tasks in the input graph to the fastest processor and writes the
        partitioning results to a file.

        Args:
            input_dir (str): The directory containing the input graph.
            output_dir (str): The directory where the partitioning results will be written.
            graph_file (str): The name of the input graph file.
            physical_graph_dir (str): The directory containing the physical graph.
            physical_graph_path (str): The path to the physical graph file.
            cr_max (float): The maximum allowed cut ratio.

        Returns:
            None
        """
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'simple_part')
        start_time = time.time()

        partition = self.simple_part(weighted_graph, physical_graph)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, cr_max, start_time)

    def write_metis_with_pg(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
    ) -> None:
        weighted_graph = input_graph(join(input_dir, graph_file))
        unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))

        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'metis_with_pg')

        start_time = time.time()
        weighted_partition = self.do_metis_with_pg(weighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, start_time)

        start_time = time.time()
        unweighted_partition = self.do_metis_with_pg(unweighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph, start_time)

    def run_from_paths(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
        check_cache: bool,
        seed: int | None,
    ) -> None:
        """
        Runs the partitioning algorithm on the given graph and writes the results to a file.

        Args:
            input_dir (str): The directory containing the input graph.
            output_dir (str): The directory where the partitioning results will be written.
            graph_file (str): The name of the input graph file.
            physical_graph_dir (str): The directory containing the physical graph.
            physical_graph_path (str): The path to the physical graph file.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check if the result is already in the cache.
            seed (int | None): The seed to use for the MK algorithm.

        Returns:
            None
        """
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph: nx.Graph = input_graph(join(physical_graph_dir, physical_graph_path))

        start_time = time.time()
        initial_weighted_partition = self.do_metis_with_pg(weighted_graph, physical_graph, cr_max=cr_max, check_cache=check_cache, seed=seed)
        self.write_results(join(output_dir.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph, cr_max, start_time)
        start_time = time.time()
        weighted_partition = self.do_greed(weighted_graph, physical_graph, initial_weighted_partition, cr_max)
        self.write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, cr_max, start_time)

    def run(self, graph: nx.Graph, physical_graph: nx.Graph, cr_max: float) -> list[int] | None:
        """
        Runs the partitioning algorithm on the given graph and physical graph.

        Args:
            graph (nx.Graph): The input graph.
            physical_graph (nx.Graph): The physical graph.
            cr_max (float): The maximum allowed cut ratio.

        Returns:
            list[int] | None: The partitioning result or None if the cut ratio is not satisfied.
        """
        initial_weighted_partition = self.do_metis_with_pg(graph, physical_graph, cr_max, check_cache)
        
        partition = self.postprocessing_phase(graph, physical_graph, initial_weighted_partition, cr_max)

        return partition
