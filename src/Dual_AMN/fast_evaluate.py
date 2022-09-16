import time
import ray
import faiss
import numpy as np
from sklearn import preprocessing

ray.init()


def merge_dic(dic1, dic2):
    for k, v in dic2.items():
        dic1[k] = v
    return dic1


def mwgm_graph_tool(pairs, sim_dic):
    from graph_tool.all import Graph, max_cardinality_matching  # necessary
    if not isinstance(pairs, list):
        pairs = list(pairs)
    g = Graph()
    weight_map = g.new_edge_property("float")
    nodes_dict1 = dict()
    nodes_dict2 = dict()
    edges = list()
    for x, y in pairs:
        sim = 1.0 - sim_dic.get((x, y))
        if x not in nodes_dict1.keys():
            n1 = g.add_vertex()
            nodes_dict1[x] = n1
        if y not in nodes_dict2.keys():
            n2 = g.add_vertex()
            nodes_dict2[y] = n2
        n1 = nodes_dict1.get(x)
        n2 = nodes_dict2.get(y)
        e = g.add_edge(n1, n2)
        edges.append(e)
        weight_map[g.edge(n1, n2)] = sim
    print("graph via graph_tool", g)
    # edges=True
    res = max_cardinality_matching(g, heuristic=True, weight=weight_map, minimize=False, edges=True)
    edge_index = np.where(res.get_array() == 1)[0].tolist()
    matched_pairs = set()
    for index in edge_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def search_faiss(index, query, num, bj, batch, top_k):
    t = time.time()
    hits = [0] * len(top_k)
    mr, mrr = 0, 0
    _, index_mat = index.search(query, num)
    for i, ent_i in enumerate(batch):
        golden = ent_i
        vec = index_mat[i,]
        golden_index = np.where(vec == golden)[0]
        if len(golden_index) > 0:
            rank = golden_index[0]
            mr += (rank + 1)
            mrr += 1 / (rank + 1)
            for j in range(len(top_k)):
                if rank < top_k[j]:
                    hits[j] += 1
    print("alignment evaluating at batch {}, hits@{} = {} time = {:.3f} s ".
          format(bj, top_k, np.array(hits) / len(batch), time.time() - t))
    return np.array(hits), mrr, mr


def search_faiss_one_by_one(index, query, num, bj, batch, top_k):
    t = time.time()
    hits = [0] * len(top_k)
    mr, mrr = 0, 0
    if len(top_k) == 1:
        num = top_k[0]
    _, index_mat = index.search(query, num)
    for i, ent_i in enumerate(batch):
        golden = ent_i
        vec = index_mat[i,]
        golden_index = np.where(vec == golden)[0]
        if len(golden_index) > 0:
            rank = golden_index[0]
            mr += (rank + 1)
            mrr += 1 / (rank + 1)
            for j in range(len(top_k)):
                if rank < top_k[j]:
                    hits[j] += 1
    print("alignment evaluating at batch {}, hits@{} = {} time = {:.3f} s ".
          format(bj, top_k, np.array(hits) / len(batch), time.time() - t))
    return np.array(hits), mrr, mr


def search_faiss_hits1_one_by_one(index, query, bj, batch):
    t = time.time()
    hits = 0
    _, index_mat = index.search(query, 1)
    for i, ent_i in enumerate(batch):
        if index_mat[i, 0] == ent_i:
            hits += 1
    # print("evaluating at batch {}, hits@1 = {} time = {:.3f} s ".format(bj, hits / len(batch), time.time() - t))
    return hits


def test_by_faiss_batch(embeds2, embeds1, top_k=[1, 5, 10], is_norm=True, batch_num=16):
    start = time.time()
    if is_norm:
        embeds1 = preprocessing.normalize(embeds1)
        embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]
    hits = np.array([0] * len(top_k))
    mr, mrr = 0, 0
    index = faiss.IndexFlatL2(dim)  # build the index
    index.add(embeds1)  # add vectors to the index
    batches = task_divide(list(range(num)), batch_num)
    query_num = 0

    rest = []
    for bj, batch in enumerate(batches):
        query_num += len(batch)
        query = embeds2[batch, :]
        rest.append(search_faiss(index, query, embeds1.shape[0], bj, batch, top_k))
    for hits_, mrr_, mr_ in rest:
        hits += hits_
        mrr += mrr_
        mr += mr_
    mr /= num
    mrr /= num
    hits = hits / num
    mr = round(mr, 8)
    mrr = round(mrr, 8)
    for i in range(len(hits)):
        hits[i] = round(hits[i], 8)
    print("alignment results with faiss: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, total time = {:.3f} s ".
          format(top_k, hits, mr, mrr, time.time() - start))
    return hits, mrr, mr


def test_by_faiss_batch_one_by_one(embeds2, embeds1, top_k=[1, 5, 10, 50], is_norm=True, batch_num=16):
    start = time.time()
    if is_norm:
        embeds1 = preprocessing.normalize(embeds1)
        embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]
    hits = np.array([0] * len(top_k))
    mr, mrr = 0, 0
    index = faiss.IndexFlatL2(dim)  # build the index
    index.add(embeds1)  # add vectors to the index
    batches = task_divide(list(range(num)), batch_num)
    query_num = 0

    for bj, batch in enumerate(batches):
        query_num += len(batch)
        query = embeds2[batch, :]
        hits_, mrr_, mr_ = search_faiss_one_by_one(index, query, embeds1.shape[0], bj, batch, top_k)
        hits += hits_
        mrr += mrr_
        mr += mr_
    mr /= num
    mrr /= num
    hits = hits / num
    mr = round(mr, 8)
    mrr = round(mrr, 8)
    for i in range(len(hits)):
        hits[i] = round(hits[i], 8)
    print("alignment results with faiss: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, total time = {:.3f} s ".
          format(top_k, hits, mr, mrr, time.time() - start))
    return hits, mrr, mr


def test_by_faiss_batch_hits1_one_by_one(embeds2, embeds1, is_norm=True, batch_num=16):
    start = time.time()
    if is_norm:
        embeds1 = preprocessing.normalize(embeds1)
        embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]
    hits = 0
    index = faiss.IndexFlatL2(dim)  # build the index
    index.add(embeds1)  # add vectors to the index
    batches = task_divide(list(range(num)), batch_num)
    query_num = 0

    for bj, batch in enumerate(batches):
        query_num += len(batch)
        query = embeds2[batch, :]
        hits_ = search_faiss_hits1_one_by_one(index, query, bj, batch)
        hits += hits_

    hits = hits / num

    hits = round(hits, 8)
    print("Hits@1 = {}, total time = {:.3f} s ".format(hits, time.time() - start))
    return hits


@ray.remote
def search_faiss_prf_one_by_one_ray(index, query, batch, n):
    rest = set()
    _, index_mat = index.search(query, n)
    for i, ent_i in enumerate(batch):
        for j in range(n):
            rest.add((ent_i, index_mat[i, j]))
    return rest


def search_faiss_prf_one_by_one(index, query, batch, n):
    rest = set()
    sim_dic = dict()
    sim_mat, index_mat = index.search(query, n)
    for i, ent_i in enumerate(batch):
        for j in range(n):
            rest.add((ent_i, index_mat[i, j]))
            sim_dic[(ent_i, index_mat[i, j])] = sim_mat[i, j]
    return (rest, sim_dic)


def test_by_faiss_batch_prf_one_by_one(embeds2, embeds1, is_norm=True, batch_num=8, if_ray=False):
    tt = time.time()
    if is_norm:
        embeds1 = preprocessing.normalize(embeds1)
        embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]

    index = faiss.IndexFlatL2(dim)  # build the index
    index.add(embeds1)  # add vectors to the index
    batches = task_divide(list(range(num)), batch_num)
    query_num = 0

    align_pairs = set()
    sim_dic = dict()
    rests = list()

    knn = 5

    if if_ray:
        for bj, batch in enumerate(batches):
            query_num += len(batch)
            query = embeds2[batch, :]
            rests.append(search_faiss_prf_one_by_one_ray.remote(index, query, batch, knn))
        for rest in ray.get(rests):
            align_pairs = align_pairs | rest
    else:
        for bj, batch in enumerate(batches):
            query_num += len(batch)
            query = embeds2[batch, :]
            rests.append(search_faiss_prf_one_by_one(index, query, batch, knn))
        for rest, dic in rests:
            align_pairs = align_pairs | rest
            sim_dic = merge_dic(sim_dic, dic)

    print("alignment search time = {:.3f} s ".format(time.time() - tt))
    return align_pairs, sim_dic


def compute_prf(correct_source_num, rest_output, rev_rest_output):
    rest, sim_dic1 = rest_output
    rev_rests, sim_dic2 = rev_rest_output
    # rev_rests = [(j, i) for i, j in rev_rests]
    inter_rest = set(rest) & set(rev_rests)

    ground_truth = set([(i, i) for i in range(correct_source_num)])
    correct_align = ground_truth & inter_rest

    p = len(correct_align) / len(inter_rest)
    r = len(correct_align) / correct_source_num
    f1 = 2 * (p * r) / (p + r)
    print("Bi-top quick results: precision = {:.3f}, recall = {:.3f}, f1-score = {:.3f}.".format(p, r, f1))

    rev_rests = [(j, i) for i, j in rev_rests]
    sim_dic3 = dict()
    for k, v in sim_dic2.items():
        sim_dic3[(k[1], k[0])] = v
    sim_dic1 = merge_dic(sim_dic1, sim_dic3)

    final_rest = mwgm_graph_tool(list(rev_rests + list(rest)), sim_dic1)
    print("final_rest", len(final_rest), len(list(rev_rests + list(rest))))
    ground_truth = set([(i, i) for i in range(correct_source_num)])
    correct_align = ground_truth & final_rest

    p = len(correct_align) / len(final_rest)
    r = len(correct_align) / correct_source_num
    f1 = 2 * (p * r) / (p + r)
    print("quick results: precision = {:.3f}, recall = {:.3f}, f1-score = {:.3f}.".format(p, r, f1))
    return p, r, f1, final_rest
