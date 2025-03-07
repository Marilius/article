from .main import MKPartitioner


graph_dirs = [
    # './data/testing_graphs',
    # './data/triangle/graphs',
    # './data/sausages',
    # './data/rand',
    './data/gen_data',
]

mk = MKPartitioner()

# mk.research()

# g_path = './data/sausages/dagP38.txt'
# G_weighted = input_networkx_graph_from_file(g_path)
# mk.CUT_RATIO = 0.3
# (n, weighted_partition) = mk.mk_part(G_weighted)
# for n_parts in [10, 15, 20, 25]:
#     ufactor = 1
#     while ufactor < 10e2:
#         weighted_partition = mk.do_metis(G_weighted, n_parts, ufactor)
#         if mk.check_cut_ratio(G_weighted, weighted_partition) and abs(len(set(weighted_partition)) - n_parts) <= 3:
#             print(ufactor, mk.CUT_RATIO, len(set(weighted_partition)), [weighted_partition.count(i) for i in sorted(list(set(weighted_partition)))])
#             break
#         ufactor += min(10, ufactor)
# mk.