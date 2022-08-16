import torch
import torch.nn as nn
import random
from .encoder import GNN_Encoder
from .relation import MLP,ContextMLP, TaskAwareRelation
from ..datasets import mol_to_graph_data_obj_simple
from rdkit import Chem
from rdkit.Chem import AllChem
from grakel import Graph
from grakel.kernels import ShortestPath
import numpy as np
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.data import DataLoader


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


def edge_list(nodes, edges, batchs):
    nodes = nodes.cpu().numpy()
    edges = edges.cpu().numpy()
    batchs = batchs.cpu().numpy()
    node_batch = {}
    batch_node_idx = {}
    for n in range(len(nodes)):
        node_batch[n] = batchs[n]
        if batchs[n] not in batch_node_idx:
            batch_node_idx[batchs[n]] = {}
        batch_node_idx[batchs[n]][n] = len(batch_node_idx[batchs[n]])
    #print(node_batch)
    #print(batch_node_idx)
    graph_edges = []
    for i in range(len(batch_node_idx)):
        graph_edges.append([])
    for i in range(len(edges[0])):
        #print(e)
        n1, n2 = edges[0][i], edges[1][i]
        graph1, graph2 = node_batch[n1], node_batch[n2]
        if graph1 != graph2:
            print("warning")
            continue
        idx1 = batch_node_idx[graph1][n1]
        idx2 = batch_node_idx[graph2][n2]
        graph_edges[graph1].append((idx1, idx2))
    #print(graph_edges)
    return graph_edges

def construct_knn(kernel_idx):
    edge_index = [[], []]
    for i in range(len(kernel_idx)):
        for j in range(len(kernel_idx[i])):
            edge_index[0].append(i)
            edge_index[1].append(kernel_idx[i, j].item())

            edge_index[0].append(kernel_idx[i, j].item())
            edge_index[1].append(i)
    return edge_index

def remove_edge(edge_index, edge_attr, drop_ratio):
    edges, attrs = dropout_adj(edge_index, edge_attr=edge_attr, p = drop_ratio)

    return edges, attrs

def drop_node(x, drop_ratio):
    node_num, _ = x.size()
    drop_num = int(node_num * drop_ratio)
    idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()
    x[idx_mask] = 0

    return x

def domain_aug(row_idx, smiles_list, rule_indicator, rules, aug_times=1):
    #row_idx = data.id.cpu().numpy()[0]
    #print(row_idx)
    s = smiles_list[row_idx]
    mol_obj = Chem.MolFromSmiles(s)
    
    mol_prev = mol_obj
    mol_next = None
    for time in range(aug_times):
        #print('aug time: ', time)
        non_zero_idx = list(np.where(rule_indicator[row_idx, :]!=0)[0])
        cnt = -1
        while len(non_zero_idx)!=0:
            col_idx = random.choice(non_zero_idx)

            # calculate counts
            rule = rules[col_idx]
            rxn = AllChem.ReactionFromSmarts(rule['smarts'])
            products = rxn.RunReactants((mol_prev,))

            cnt = len(products)
            if cnt != 0:
                break
            else:
                non_zero_idx.remove(col_idx)
        
        if cnt >= 1:
            aug_idx = random.choice(range(cnt))
            mol = products[aug_idx][0]
            try:
                Chem.SanitizeMol(mol)
            except: # TODO: add detailed exception
                pass

            mol_next = mol
            mol_prev = mol
            #rule_indicator[row_idx, col_idx] -= 1
        else:
            mol_next = mol_prev
    assert mol_next
    data = mol_to_graph_data_obj_simple(mol_next)
    return data

def loader_to_samples(data):
    loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
    for samples in loader:
        samples=samples.cuda()
        return samples
    
def aug_rule(data, smiles_list, rule_indicator, rules):
    aug_data = []
    for d in data.id.cpu().numpy():
        aug_data.append(domain_aug(d, smiles_list, rule_indicator, rules))
    aug_data = loader_to_samples(aug_data)
    return aug_data

class ContextAwareRelationNet(nn.Module):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        self.rel_layer = args.rel_layer
        self.edge_type = args.rel_adj
        self.edge_activation = args.rel_act
        self.gpu_id = args.gpu_id
        self.aug_num = args.aug_num
        self.aug_ratio = args.drop_ratio
        self.aug = args.aug
        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)


        self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                batch_norm=args.batch_norm,dropout=args.map_dropout,
                                pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)

        inp_dim = args.map_dim
        self.k = args.rel_k
        self.task_graph = {}
        self.adapt_relation = TaskAwareRelation(inp_dim=inp_dim, hidden_dim=args.rel_hidden_dim,
                                                num_layers=args.rel_layer, edge_n_layer=args.rel_edge_layer,
                                                node_n_layer=args.node_n_layer,
                                                top_k=args.rel_k, res_alpha=args.rel_res,
                                                batch_norm=args.batch_norm, adj_type=args.rel_adj,
                                                activation=args.rel_act, node_concat=args.rel_node_concat,dropout=args.rel_dropout,
                                                pre_dropout=args.rel_dropout2,model_file=args.pretrained_weight_path,gpu_id=self.gpu_id)
        self.rules = None
        self.smiles_list = None
        self.rule_indicator = None

    def to_one_hot(self,class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand
        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge=edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False,return_adj=False,return_emb=False, adj=None):
        if not return_emb:
            s_logits, q_logits, adj, task_feat = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb, adj=adj)
        else:
            s_logits, q_logits, adj, s_rel_emb, q_rel_emb, task_feat = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb, adj=adj)
        if q_pred_adj:
            q_sim = adj[-1][:, 0, -1, :-1]
            q_logits = q_sim @ self.to_one_hot(s_label)
        if not return_emb:
            return s_logits, q_logits, adj, task_feat
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb, task_feat
    
    def construct_graph_old(self, s_data, q_data):
        graph_edges_s = edge_list(s_data.x, s_data.edge_index, s_data.batch)
        graph_edges_q = edge_list(q_data.x, q_data.edge_index, q_data.batch)
        n_query = len(graph_edges_q)
        adjs = []
        for i in range(n_query):
            graph_edges = graph_edges_s + [graph_edges_q[i]]
            n_graph = len(graph_edges)
            Gs = []
            for g in graph_edges:
                if g == []:
                    g = [(0, 0)]
                G = Graph(g)
                Gs.append(G)
            gk = ShortestPath(normalize=True, with_labels=False)
            kernel_simi = torch.tensor(gk.fit_transform(Gs))
            kernel_idx = torch.topk(kernel_simi, k=self.k, dim=1, largest=True)[1][:, 1:]
            
            adj = np.zeros((n_graph, n_graph))
            for j in range(len(kernel_idx)):
                for k in kernel_idx[j]:
                    '''
                    if j < self.k:
                        for i in range(self.k):
                            adj[j][i] = kernel_simi[j][i].item()
                    elif j >= self.k and j < 2 * self.k:
                        for i in range(self.k, 2 * self.k):
                            adj[j][i] = kernel_simi[j][i].item()
                    '''
                    adj[j][k.item()] = kernel_simi[j][k.item()].item()
            
            #adj = np.array(gk.fit_transform(Gs))
            adj[np.isnan(adj)] = 0
            #print(adj)
            adjs.append(adj)
            #print(adj.shape)
        kernel_adj = torch.tensor(np.array(adjs),dtype=torch.float32).cuda()
        #self.task_graph[task_id] = kernel_adj
        return kernel_adj
    
    def construct_graph_old2(self, s_data, q_data, kernel):
        n_query = len(q_data.id)
        n_data = len(s_data.id) + 1
        s_idxs = s_data.id.reshape(1, -1).repeat(n_query, 1)
        q_idxs = q_data.id.reshape(-1, 1)
        idxs = torch.cat((s_idxs, q_idxs), 1).reshape(-1, 1).squeeze(1).cpu().tolist()
        tmp_adjs = kernel[np.ix_(idxs, idxs)]
        #print(tmp_adjs.shape)
        adj = []
        for i in range(n_query):
            adj.append(tmp_adjs[n_data * i:n_data * (i + 1), n_data * i:n_data * (i + 1)])
        adjs = np.array(adj)
        #print(adjs.shape)
        #print(adjs)
        #adjs[np.isnan(adjs)] = 0
        #print(np.array(adjs))                
        kernel_adj = torch.tensor(adjs,dtype=torch.float32).cuda()
        return kernel_adj
        
    def construct_graph_old3(self, kernel_simis):
        adjs = []
        for kernel_simi in kernel_simis:
            n_graph = kernel_simi.size(0)
            #print(kernel_simi)
            kernel_idx = torch.topk(kernel_simi, k=self.k, dim=1, largest=True)[1][:, 1:]           
            adj = np.zeros((n_graph, n_graph))
            for j in range(len(kernel_idx)):
                '''
                if j < self.k:
                    for i in range(self.k):
                        adj[j][i] = kernel_simi[j][i].item()
                elif j >= self.k and j < 2 * self.k:
                    for i in range(self.k, 2 * self.k):
                        adj[j][i] = kernel_simi[j][i].item()
                '''
                
                for k in kernel_idx[j]:
                    adj[j][k.item()] = kernel_simi[j][k.item()].item()
            adj[np.isnan(adj)] = 0
            #print(adj)
            adjs.append(adj)
            #print(adj.shape)
        kernel_adj = torch.tensor(np.array(adjs),dtype=torch.float32).cuda()
        #self.task_graph[task_id] = kernel_adj
        return kernel_adj
    
    def construct_graph(self, kernel_simis):
        a,_= kernel_simis.topk(k=self.k, dim=2)
        a_min = torch.min(a,dim=-1).values
        n_query, n_node, n_node = kernel_simis.size()
        a_min = a_min.unsqueeze(-1).repeat(1, 1, n_node)
        ge = torch.ge(kernel_simis, a_min)
        zero = torch.zeros_like(kernel_simis)
        kernel_adj = torch.where(ge, kernel_simis, zero)
        #kernel_adj = torch.tensor(np.array(adjs),dtype=torch.float32).cuda()
        #self.task_graph[task_id] = kernel_adj
        return kernel_adj
    
    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False, task_id=None, kernel_simi=None):
        #kernel_adj = self.construct_graph(s_data, q_data)
        #print(q_data.id)
        #return "done"
        #if kernel_simi is None:
        if len(kernel_simi.shape) == 2:
            kernel_simi_new = self.construct_graph_old2(s_data, q_data, kernel_simi)
            #return "done"
            kernel_adj = self.construct_graph(kernel_simi_new)
        else:
            kernel_adj = self.construct_graph(kernel_simi)
        graph_augs = []
        for i in range(self.aug_num):
            if self.aug == 'RE':
                graph_aug = remove_edge(s_data.edge_index, s_data.edge_attr, self.aug_ratio)
            elif self.aug == 'rule':
                graph_aug = aug_rule(s_data, self.smiles_list, self.rule_indicator, self.rules)
            else:
                graph_aug = drop_node(s_data.x, self.aug_ratio)
            graph_augs.append(graph_aug)
        
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

        s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
        s_emb_map_augs = []
        s_emb_augs = [s_emb]
        for i in range(self.aug_num):
            if self.aug == 'RE':
                #print(s_data.edge_index)
                #print(graph_aug[i])
                s_emb_aug, _ = self.mol_encoder(s_data.x, graph_augs[i][0], graph_augs[i][1], s_data.batch)
            elif self.aug == 'rule':
                aug_data = graph_augs[i]
                s_emb_aug, _ = self.mol_encoder(aug_data.x, aug_data.edge_index, aug_data.edge_attr, aug_data.batch)
            else:
                s_emb_aug, _ = self.mol_encoder(graph_augs[i], s_data.edge_index, s_data.edge_attr, s_data.batch)
            s_emb_augs.append(s_emb_aug)
            s_emb_map_aug, q_emb_map = self.encode_projection(s_emb_aug, q_emb)
            s_emb_map_augs.append(s_emb_map_aug)
        s_logits, q_logits, adj, task_feat = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj, adj=kernel_adj)
        s_logits_augs, q_logits_augs = [], []
        for i in range(self.aug_num):
            s_logits_aug, q_logits_aug, _, _ = self.relation_forward(s_emb_map_augs[i], q_emb_map, s_label, q_pred_adj=q_pred_adj, adj=kernel_adj)
            s_logits_augs.append(s_logits_aug)
            q_logits_augs.append(q_logits_aug)
        s_logits_augs.append(s_logits)
        return s_logits, q_logits, adj, s_node_emb, s_logits_augs, q_logits_augs, task_feat, s_emb_augs

    def forward_query_list(self, s_data, q_data_list, s_label=None, q_pred_adj=False):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb_list = [self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)[0] for q_data in
                      q_data_list]

        q_logits_list, adj_list = [], []
        for q_emb in q_emb_list:
            s_emb_map,q_emb_map = s_emb, q_emb
            #s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit.detach())
            if adj is not None:
                sim_adj = adj[-1][:,0].detach()
                q_adj = sim_adj[:,-1]
                adj_list.append(q_adj)

        q_logits = torch.cat(q_logits_list, 0)
        adj_list = torch.cat(adj_list, 0)
        return s_logit.detach(),q_logits, adj_list

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False, task_id=None, kernel_simi=None):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        y_true_list=[]
        q_logits_list, adj_list = [], []
        #kernel_adj = self.construct_graph(kernel_simi)
        for q_data in q_loader:
            q_data = q_data.to(s_emb.device)
            #print(q_data.id)
            #return "done"
            y_true_list.append(q_data.y)
            kernel_simi_new = self.construct_graph_old2(s_data, q_data, kernel_simi)
            kernel_adj = self.construct_graph(kernel_simi_new)
            graph_augs = []
            for i in range(self.aug_num):
                if self.aug == 'RE':
                    graph_aug = remove_edge(s_data.edge_index, s_data.edge_attr, self.aug_ratio)
                else:
                    graph_aug = drop_node(s_data.x, self.aug_ratio)
                graph_augs.append(graph_aug)
            q_emb,_ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_emb_map_augs = []
            for i in range(self.aug_num):
                if self.aug == 'RE':
                    s_emb_aug, _ = self.mol_encoder(s_data.x, graph_augs[i][0], graph_augs[i][1], s_data.batch)
                else:
                    s_emb_aug, _ = self.mol_encoder(graph_augs[i], s_data.edge_index, s_data.edge_attr, s_data.batch)
                s_emb_map_aug, q_emb_map = self.encode_projection(s_emb_aug, q_emb)
                s_emb_map_augs.append(s_emb_map_aug)
            s_logit, q_logit, adj, _ = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj, adj=kernel_adj)
            s_logits_augs, q_logits_augs = [], [q_logit]
            for i in range(self.aug_num):
                s_logits_aug, q_logits_aug, _, _ = self.relation_forward(s_emb_map_augs[i], q_emb_map, s_label, q_pred_adj=q_pred_adj, adj=kernel_adj)
                s_logits_augs.append(s_logits_aug)
                q_logits_augs.append(q_logits_aug)
            s_logits_augs.append(s_logit)
            #s_logit = torch.stack(s_logits_augs).mean(dim=0)
            #q_logit = torch.stack(q_logits_augs).mean(dim=0)
            q_logits_list.append(q_logit)
            if adj is not None:
                sim_adj = adj[-1].detach()
                adj_list.append(sim_adj)

        q_logits = torch.cat(q_logits_list, 0)
        y_true = torch.cat(y_true_list, 0)
        sup_labels={'support':s_data.y,'query':y_true_list}
        return s_logit, q_logits, y_true,adj_list,sup_labels, s_logits_augs