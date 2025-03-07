from copy import deepcopy

import argparse

import random
import itertools

from dataclasses import dataclass


@dataclass
class Job:
    id: int
    proc: int
    length: int
    start_time: float
    end_time: float
    edges: list[int]


class RandomException(Exception):
    """
    Превышено число попыток нарандомить.
    """
    pass


parser = argparse.ArgumentParser(description='Генератор графа по параметрам.')

parser.add_argument('-p', nargs="+", type=int, help='Производительности процессоров')
parser.add_argument('-L', type=int, help='Суммарная длительность работ на процессоре, одинаковая для каждого процессора.')
parser.add_argument('-min_l', type=int, help='Минимальная длительность работ на процессоре.')
parser.add_argument('-max_l', type=int, help='Максимальная длительность работ на процессоре.')
parser.add_argument('-N', type=float, help='Число рёбер на вершину.')
parser.add_argument('-cr', type=float, help='Доля секущих рёбер.')
parser.add_argument('-n_tries', type=int, help='Число попыток нарандомить.', default=100)
parser.add_argument('--shuffle_off', dest='shuffle', action='store_false', help='Перемешивание номеров вершин.', default=True)

WRITE_UNSHUFFLED = True

# парсим командную строку
args = parser.parse_args()
p: list[int] = args.p
L: int = args.L
min_l: int = args.min_l
max_l: int = args.max_l
N: float = args.N
cr: float = args.cr
n_tries: int = args.n_tries
shuffle: bool = args.shuffle

# создание вершин графа
n0: int = 0
jobs: list[list[Job]] = [[] for _ in range(len(p))]
for i in range(len(p)):
    f = True
    while f:
        f = False
        n = n0
        time_left = L * p[i]
        start_time = 0
        while time_left > 0:
            curr_time = random.randint(min_l, max_l)
            end_time = min(start_time + curr_time, L * p[i])
            curr_time = end_time - start_time
            
            if n_tries <= 0:
                raise RandomException()

            if curr_time < min_l:
                n_tries -= 1
                f = True
                jobs[i] = []
                break

            jobs[i].append(Job(n, i, curr_time, start_time, end_time, []))
            n += 1

            start_time += curr_time
            time_left -= curr_time
            assert time_left >= 0
            assert min_l <= curr_time <= max_l

    n0 = n

# число рёбер и число секущих рёбер
N_e: int = int(n0 * N) 
N_s: int = int(N_e * cr)

exact_partition: list[int] = list(itertools.chain.from_iterable([[proc] * len(job_list) for proc, job_list in enumerate(jobs)]))
assert len(exact_partition) == len(list(itertools.chain.from_iterable(jobs)))

for i in range(len(p)):
    for job in jobs[i]:
        assert min_l <= job.length <= max_l
        assert 0 <= job.start_time < job.end_time <= L * p[i]

for i, proc_jobs in enumerate(jobs):
    weight = 0
    for job in proc_jobs:
        weight += job.length
    assert weight == L * p[i], f'{weight} != {L * p[i]}'

# добавление не секущих рёбер
n = N_e - N_s
edge_list: list[list[int]] = [[] for _ in range(sum(map(len, jobs)))]
while n:
    proc_num = random.randint(0, len(jobs) - 1)
    
    while (first := random.choice(jobs[proc_num]).id) == (second := random.choice(jobs[proc_num]).id):
        if n_tries <= 0:
            raise RandomException()
        n_tries -= 1
    
    first, second = min(first, second), max(first, second)

    if second not in edge_list[first]:
        assert first < second
        edge_list[first].append(second)
        n -= 1

n_all = 0
for i in jobs:
    n = 0
    for j in i:
        n += len(edge_list[j.id])
    n_all += n

assert n_all == N_e - N_s, f'{n_all} != {N_e - N_s}'


# добавление секущих рёбер
n = N_s
while n:
    while (proc_first := random.randint(0, len(jobs) - 1)) == (proc_second := random.randint(0, len(jobs) - 1)):
        if n_tries <= 0:
            raise RandomException()
        n_tries -= 1
    
    first_job = random.choice(jobs[proc_first]) 
    second_job = random.choice(jobs[proc_second])
    
    if first_job.end_time / p[proc_first] <= second_job.start_time / p[proc_second]:
        first, second = first_job.id, second_job.id
    elif second_job.end_time / p[proc_second] <= first_job.start_time / p[proc_first]:
        first, second = second_job.id, first_job.id
    else:
        continue

    if second not in edge_list[first]:
        edge_list[first].append(second)
        n -= 1

n_all = 0
for i in jobs:
    n = 0
    for j in i:
        n += len(edge_list[j.id])
    n_all += n
assert n_all == N_e

if WRITE_UNSHUFFLED and shuffle:
    initial_jobs = deepcopy(jobs)
    initial_exact_partition = deepcopy(exact_partition)
    initial_edge_list = deepcopy(edge_list)

if shuffle:
    jobs_ids = [job.id for job in itertools.chain.from_iterable(jobs)]
    zipped = list(zip(jobs_ids, exact_partition, edge_list))
    random.shuffle(zipped)
    jobs_ids, exact_partition, edge_list = zip(*zipped)

    for job in itertools.chain.from_iterable(jobs):
        job.id = jobs_ids.index(job.id)
    for job_edges in edge_list:
        for i in range(len(job_edges)):
            job_edges[i] = jobs_ids.index(job_edges[i])
    assert len(set([job.id for job in itertools.chain.from_iterable(jobs)])) == len(list(itertools.chain.from_iterable(jobs)))

    weights = [0] * len(p)
    fact_jobs = [[] for _ in range(len(p))]
    job_list = list(itertools.chain.from_iterable(jobs))
    for i, proc in enumerate(exact_partition):
        for job in job_list:
            if job.id == i:
                weights[proc] += job.length
                fact_jobs[proc].append(job.id)
                break
    for i, w in enumerate(weights):
        assert w == L * p[i]

# запись в файл
GRAPH_NAME_FORMAT = './data/gen_data/{p}_{L}_{min_l}_{max_l}_{N}_{cr}_{shuffle}.graph'
GRAPH_FORMAT = '{p}\n{L}\n{min_l} {max_l}\n{N_e} {N_s}\n'
NODE_FORMAT = '{id} {weight} {child_list}\n'
PARTITION_NAME_FORMAT = './data/gen_data/{p}_{L}_{min_l}_{max_l}_{N}_{cr}_{shuffle}.partition'
PARTITION_FORMAT = '{exact_partition}\n'

name = GRAPH_NAME_FORMAT.format(
    p='_'.join(map(str, p)),
    L=L,
    min_l=min_l,
    max_l=max_l,
    N=N,
    cr=cr,
    shuffle=shuffle,
)

with open(name, 'w+') as f:
    f.write(
        GRAPH_FORMAT.format(
            p=' '.join(map(str, p)),
            L=L,
            min_l=min_l,
            max_l=max_l,
            N_e=N_e,
            N_s=N_s,
        )
    )

    lines = []

    for proc_jobs in jobs:
        for job in proc_jobs:
            lines.append(
                (
                    job.id,
                    job.length,
                    ' '.join(map(str, sorted(edge_list[job.id])))
                )
            )

    lines.sort(key=lambda x: x[0])
    for line in lines:
        line = dict(zip(('id', 'weight', 'child_list'), line))
        f.write(NODE_FORMAT.format(**line))
print(name)

name = PARTITION_NAME_FORMAT.format(
    p='_'.join(map(str, p)),
    L=L,
    min_l=min_l,
    max_l=max_l,
    N=N,
    cr=cr,
    shuffle=shuffle,
)


with open(name, 'w+') as f:
    f.write(PARTITION_FORMAT.format(
            exact_partition=' '.join(map(str, exact_partition))
        )
    )

if WRITE_UNSHUFFLED and shuffle:
    name = GRAPH_NAME_FORMAT.format(
        p='_'.join(map(str, p)),
        L=L,
        min_l=min_l,
        max_l=max_l,
        N=N,
        cr=cr,
        shuffle=False,
    )

    with open(name, 'w+') as f:
        f.write(
            GRAPH_FORMAT.format(
                p=' '.join(map(str, p)),
                L=L,
                min_l=min_l,
                max_l=max_l,
                N_e=N_e,
                N_s=N_s,
            )
        )

        lines = []

        for proc_jobs in initial_jobs:
            for job in proc_jobs:
                lines.append(
                    (
                        job.id,
                        job.length,
                        ' '.join(map(str, sorted(initial_edge_list[job.id])))
                    )
                )

        lines.sort(key=lambda x: x[0])
        for line in lines:
            line = dict(zip(('id', 'weight', 'child_list'), line))
            f.write(NODE_FORMAT.format(**line))

    name = PARTITION_NAME_FORMAT.format(
        p='_'.join(map(str, p)),
        L=L,
        min_l=min_l,
        max_l=max_l,
        N=N,
        cr=cr,
        shuffle=False,
    )

    with open(name, 'w+') as f:
        f.write(PARTITION_FORMAT.format(
                exact_partition=' '.join(map(str, initial_exact_partition))
            )
        )
