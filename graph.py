import os
import sys
import shutil
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
import time
from tqdm import tqdm
import pickle
from xtract import xtract


class DisJointSets:
    def __init__(self, N):
        self.N = N
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

        self._edges = []

    def get_wcc(self):
        wcc = {}
        for node in range(self.N):
            root = self.find(node)
            if root not in wcc:
                wcc[root] = set()
                wcc[root].add(root)
            wcc[root].add(node)

        wcc_edges = {}
        for n1, n2, attr in self._edges:
            r = self.find(n1)
            assert(n2 in wcc[r])

            if r not in wcc_edges:
                wcc_edges[r] = set()
            wcc_edges[r].add((n1, n2, tuple(attr)))

        return wcc, wcc_edges

    def find(self, u):
        assert(u < self.N)

        while u != self._parents[u]:
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def union(self, u, v, attr):
        assert(u < self.N and v < self.N)

        self._edges.append((u, v, attr))

        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True
        
        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False


def gen_pt(cnf_dir_path, pt_dir_path, n_cpu=1):

    if not os.path.isdir(pt_dir_path):
        os.makedirs(pt_dir_path)

    task_lst = []
    for cnf_name in sorted(os.listdir(cnf_dir_path)):
        cnf_path = cnf_dir_path + "/" + cnf_name
        if os.path.isfile(cnf_path):
            if (cnf_path.endswith(".xz") or \
                cnf_path.endswith(".bz2") or \
                cnf_path.endswith(".lzma") or \
                cnf_path.endswith(".gz")) and \
                not os.path.isfile(pt_dir_path + "/" + cnf_name + ".c-0.pt"):
                task_lst.append([cnf_dir_path, cnf_name, pt_dir_path])
    
    with Pool(n_cpu) as p:
        with tqdm(total=len(task_lst)) as pbar:
            for i, _ in enumerate(p.imap_unordered(gen_pt_single, task_lst)):
                pbar.update()
    
    print("Parallel Extraction Finished")


def gen_pt_single(arg_lst):
    cnf_dir_path, cnf_name, pt_dir_path = arg_lst[0], arg_lst[1], arg_lst[2]
    cnf_path = cnf_dir_path + "/" + cnf_name

    backbone_name = cnf_name + ".backbone.xz"
    backbone_dir_path = "./data/backbone/" + \
                        cnf_dir_path.split("/")[-1]
    backbone_path = backbone_dir_path + "/" + backbone_name

    if not os.path.isfile(backbone_path):
        backbone_name_sec = ".".join(cnf_name.split(".")[:-1]) + ".backbone.xz"
        backbone_path = backbone_dir_path + "/" + backbone_name_sec

    cnf_path = cnf_dir_path + "/" + cnf_name
    dc_cnf_path = ""
    if cnf_name.endswith(".xz") :
        dc_cnf_path = cnf_path[0:-3]        
    elif cnf_name.endswith(".gz"):
        dc_cnf_path = cnf_path[0:-3]
    elif cnf_name.endswith(".lzma"):
        dc_cnf_path = cnf_path[0:-5]
    elif cnf_name.endswith(".bz2"):
        dc_cnf_path = cnf_path[0:-4]
    else:
        print("unknown compress format: " + cnf_name)
        return
        
    if os.path.exists(dc_cnf_path):
        os.remove(dc_cnf_path)
    xtract(cnf_path, cnf_dir_path)
    
    assert(os.path.isfile(dc_cnf_path))

    dc_backbone_path = None
    if os.path.isfile(backbone_path):
        # backbone file name endswith .xz
        dc_backbone_path = backbone_path[:-3] 
        if os.path.exists(dc_backbone_path):
            os.remove(dc_backbone_path)
        xtract(backbone_path, backbone_dir_path)
    else:
        print(f"backbone file does not exist: {backbone_path}")
    
    assert(os.path.isfile(dc_cnf_path))

    data_lst, wcc = cnf_to_pt_bipartite(dc_cnf_path, dc_backbone_path)

    assert(os.path.isfile(dc_cnf_path))

    os.remove(dc_cnf_path)
    if dc_backbone_path is not None:
        os.remove(dc_backbone_path)

    if data_lst is None:
        return

    for i, data in enumerate(data_lst):
        pt_path = pt_dir_path + "/" + cnf_name + f".c-{i}.pt"
        try:
            if os.path.isfile(pt_path):
                os.remove(pt_path)
            torch.save(data, pt_path)
        except Exception as e:
            print(e)
            
            tmp_dir_path = "/".join(pt_dir_path.split("/")[:-1]) + "/tmp"
            print(f"temporarily save to {tmp_dir_path}")
            if not os.path.isdir(tmp_dir_path):
                os.makedirs(tmp_dir_path)
            tmp_path = tmp_dir_path + "/" + cnf_name + f".c-{i}.pt"
            torch.save(data, tmp_path)

def cnf_to_pt_bipartite(cnf_file_path, backbond_file_path, timelim=1000):
    start_time = time.time()

    backbone = set()
    if backbond_file_path is not None:
        with open(backbond_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    lit = int(line.split()[-1])
                    if lit != 0:
                        backbone.add(lit)

        if len(backbone) == 0:
            print(f"warning: no backbone in the file: {backbond_file_path}")
            return None, None

    X = []
    v2n = {}
    var_num = 0
    with open(cnf_file_path, "r") as f:
        for line in f:
            if time.time() - start_time > timelim:
                print("warning: timeout while reading cnf")
                return None, None

            line = line.strip()

            if len(line) == 0:
                continue

            fe = line[0]
            if fe == "c" or fe == "p":
                continue
            else:
                lit_lst = [int(lit) for lit in line.split()[:-1]]
                for lit in lit_lst:
                    var = abs(lit)
                    if var not in v2n:
                        v2n[var] = len(X)
                        X.append([1])
                        var_num += 1

    # backbone
    y = []
    if backbond_file_path is not None:
        y = [2 for _ in range(var_num)]
        for var, node_id in v2n.items():
            if var in backbone:
                assert(-var not in backbone)
                y[node_id] = 0
            elif -var in backbone:
                y[node_id] = 1

        assert(len(X) == len(y))

    # clauses
    edge_index = []
    edge_attr = []
    with open(cnf_file_path, "r") as f:
        for line in f:
            if time.time() - start_time > timelim:
                print("warning: timeout while reading cnf")
                return None, None

            line = line.strip()

            if len(line) == 0:
                continue

            fe = line[0]

            if fe == "c" or fe == "p":
                continue
            else:
                lit_lst = [int(lit) for lit in line.split()[:-1]]

                cla_node_id = len(X)
                X.append([-1]) # it is a clause # [0, 1, 0]

                for lit in lit_lst:
                    var = abs(lit)
                    var_node_id = v2n[var]

                    # to save disk space, we only save an direct edge
                    # need to be extended to undirected in Dataset.get()
                    edge_index.append([var_node_id, cla_node_id])

                    if lit > 0:
                        edge_attr.append([1]) # + backbone # [1, 0, 0]
                    else:
                        assert(lit < 0)
                        edge_attr.append([-1]) # - backbone # [0, 1, 0]

    assert(len(edge_index) == len(edge_attr))

    if len(y) > 0 and 0 not in y and 1 not in y:
        print(f"warning: no backbone in the data: {backbond_file_path}", flush=True)
        return None, None

    wcc = None
    ds = DisJointSets(len(X))
    for idx, edge in enumerate(edge_index):
        if time.time() - start_time > timelim:
            print("warning: timeout while constructing disjoint sets")
            return None, None

        from_node, to_node = edge[0], edge[1]
        ds.union(from_node, to_node, edge_attr[idx])
    wcc, wcc_edges = ds.get_wcc()
    assert(len(wcc) > 0 and len(wcc_edges) > 0)

    if time.time() - start_time > timelim:
        print("warning: timeout after solving wcc")
        return None, None

    data_lst = []
    if len(wcc) == 1:

        # add a root node
        root_node = len(X)
        for clause_node in range(var_num, len(X)):
            edge_index.append([root_node, clause_node])
            edge_attr.append([0])
        X.append([0])

        X = torch.tensor(X, dtype=torch.int8)
        edge_index = torch.tensor(edge_index, dtype=torch.int32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.int8)

        n2v = [-1 for _ in range(len(v2n))]
        for v, n in v2n.items():
            n2v[n] = v

        assert(all(e != -1 for e in n2v))
        n2v = torch.tensor(n2v, dtype=torch.int32)

        if len(y) > 0:
            y = torch.tensor(y, dtype=torch.int8)
            data = Data(x=X, n2v=n2v, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            data_lst.append(data)
        else:
            data = Data(x=X, n2v=n2v, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            data_lst.append(data)
    else:
        for root, c in wcc.items():
            if time.time() - start_time > timelim:
                print("warning: timeout while enumerating wcc")
                return None, None

            if len(c) == 1:
                continue

            c = sorted(list(c))

            old_n2new_n = {}
            for i, n in enumerate(c):
                old_n2new_n[n] = i

            var_node_cnt = 0
            for n in c:
                if n < var_num:
                    var_node_cnt += 1
            X_sub = [X[n] for n in c]
            
            y_sub = []
            if len(y) > 0:
                c_var = c[:var_node_cnt]
                y_sub = [y[n] for n in c_var]

                if 0 not in y_sub and 1 not in y_sub:
                    continue
            
            n2v_sub = [-1 for _ in range(var_node_cnt)]
            for v, n in v2n.items():
                if n in old_n2new_n:
                    n2v_sub[old_n2new_n[n]] = v
            
            edge_index_sub = []
            edge_attr_sub = []

            edges = wcc_edges[root]
            for edge in edges:
                if edge[0] not in old_n2new_n or edge[1] not in old_n2new_n:
                    print("BUG:", cnf_file_path)

                assert(edge[0] in old_n2new_n and edge[1] in old_n2new_n)
                node_a = old_n2new_n[edge[0]]
                node_b = old_n2new_n[edge[1]]
                attr = list(edge[2])
            
                edge_index_sub.append([node_a, node_b])
                edge_attr_sub.append(attr)

            if len(X_sub) <= 2:
                continue

            # add a root node
            root_node = len(X_sub)
            for clause_node in range(var_node_cnt, len(X_sub)):
                edge_index_sub.append([root_node, clause_node])
                edge_attr_sub.append([0])
            X_sub.append([0])

            X_sub = torch.tensor(X_sub, dtype=torch.int8)
            edge_index_sub = torch.tensor(edge_index_sub, dtype=torch.int32)
            edge_attr_sub = torch.tensor(edge_attr_sub, dtype=torch.int8)
            n2v_sub = torch.tensor(n2v_sub, dtype=torch.int32)
            if len(y_sub) > 0:
                y_sub = torch.tensor(y_sub, dtype=torch.int8)
                data = Data(x=X_sub, n2v=n2v_sub, y=y_sub, edge_index=edge_index_sub.t().contiguous(), edge_attr=edge_attr_sub)
                data_lst.append(data)
            else:
                data = Data(x=X_sub, n2v=n2v_sub, edge_index=edge_index_sub.t().contiguous(), edge_attr=edge_attr_sub)
                data_lst.append(data)

    if len(data_lst) == 0:
        print(f"warning: no data object in the data_lst: {backbond_file_path}", flush=True)
    return data_lst, wcc


if __name__ == '__main__':
    cnf_dir_path = "./data/cnf/" + sys.argv[1]
    if os.path.isdir(cnf_dir_path):
        pt_dir_path = "./data/pt/" + sys.argv[1] + "/processed"
        if not os.path.isdir(pt_dir_path):
            os.makedirs(pt_dir_path)
            os.makedirs("./data/pt/" + sys.argv[1] + "/raw")
        gen_pt(cnf_dir_path, pt_dir_path, n_cpu=1)
    else:
        print("ERROR: cnf directory name is missing! Please rerun this program with the directory name.")