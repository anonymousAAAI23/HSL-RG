import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader
from grakel import Graph
from grakel.kernels import ShortestPath

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger
import logging
import random
from time import time
import json
import pickle as pkl
import inspect
#from .gpu_mem_track import MemTracker

device = torch.device('cuda:0')

def consis_loss(logps):
    ps = [F.softmax(logp.reshape(-1, logp.shape[-1])) for logp in logps] 
    sum_p = 0.
    for p in ps:
        sum_p += p
    avg_p = sum_p / len(ps)
    loss = 0.
    for p in ps:
        loss += torch.mean((p - avg_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss 

def consis_loss_old(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = self.softmax(x)
        return x
    
class Attention2(nn.Module):
    def __init__(self, dim):
        super(Attention2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = F.sigmoid(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim1, out_dim2):
        super(MLP, self).__init__()
        self.transform1 = nn.Linear(in_dim, out_dim1)
        self.transform2 = nn.Linear(in_dim, out_dim2)
        self.attn = nn.MultiheadAttention(in_dim, 1)

    def forward(self, x, task_feats):
        x = x.reshape(1, 1, -1)
        #print(x.shape, task_feats.shape)
        h, w = self.attn(x, task_feats.unsqueeze(1), task_feats.unsqueeze(1))
        x = h.squeeze(0).squeeze(0)
        x1 = self.transform1(x)
        x1 = F.sigmoid(x1)
        x2 = self.transform2(x)
        x2 = F.sigmoid(x2)
        return x1, x2, w.squeeze(0).detach().cpu().numpy()

def cal_kernel(dataset):
    graph_edges = []
    #print(dataset)
    #print(dataset.data)
    #print(dataset.slices)
    for data in dataset:
        edges = []
        for i in range(len(data.edge_index[0])):
            n1, n2 = data.edge_index[0][i].item(), data.edge_index[1][i].item()
            edges.append((n1, n2))
        graph_edges.append(edges)
    Gs = []
    for g in graph_edges:
        #print(g)
        if g == []:
            g = [(0, 0)]
        G = Graph(g)
        Gs.append(G)
    gk = ShortestPath(normalize=True, with_labels=False)
    kernel_simi = np.array(gk.fit_transform(Gs))
    return kernel_simi

    
class Meta_Trainer(nn.Module):
    def __init__(self, args, model, pool=None):
        super(Meta_Trainer, self).__init__()

        self.args = args
        self.pool = pool

        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query
        self.aug_num = args.aug_num

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger
        self.logger2 = logging.getLogger(__name__)
        self.attention = Attention(args.map_dim).to(args.device)
        self.optimizer_attn = optim.AdamW(self.attention.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.attention2 = Attention2(args.map_dim * 2).to(args.device)
        self.optimizer_attn2 = optim.AdamW(self.attention2.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.mlp = MLP(args.map_dim * 2, args.map_dim * 2, 2).to(args.device)
        self.optimizer_mlp = optim.AdamW(self.mlp.parameters(), lr=args.mlp_lr, weight_decay=args.weight_decay)

        preload_train_data = {}
        train_kernel = {}
        if not os.path.exists(args.dataset+"_kernel"):
            os.mkdir(args.dataset+"_kernel")
        self.rules = json.load(open('isostere_transformations_new.json'))
        self.train_rules = {}
        self.test_rules = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
                if not os.path.exists(args.dataset+"_kernel/kernel_" + str(task) + ".npy"):
                    kernel_simi = cal_kernel(dataset)
                    np.save(args.dataset+"_kernel/kernel_" + str(task) + ".npy", kernel_simi)
                else:
                    #kernel_simi = cal_kernel(dataset)
                    kernel_simi = np.load(args.dataset+"_kernel/kernel_" + str(task) + ".npy")
                train_kernel[task] = kernel_simi
                with open('rules/'+ self.dataset + '/' + str(task + 1) + '/rule_indicator_new.pkl', 'rb') as f:
                    d = pkl.load(f)
                    rule_indicator = d[0]
                self.train_rules[task] = rule_indicator
        preload_test_data = {}
        test_kernel = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
                if not os.path.exists(args.dataset+"_kernel/kernel_" + str(task) + ".npy"):
                    kernel_simi = cal_kernel(dataset)
                    np.save(args.dataset+"_kernel/kernel_" + str(task) + ".npy", kernel_simi)
                else:
                    kernel_simi = np.load(args.dataset+"_kernel/kernel_" + str(task) + ".npy")
                test_kernel[task] = kernel_simi
                with open('rules/'+ self.dataset + '/' + str(task + 1) + '/rule_indicator_new.pkl', 'rb') as f:
                    d = pkl.load(f)
                    rule_indicator = d[0]
                self.test_rules[task] = rule_indicator
        self.preload_train_data = preload_train_data
        self.train_kernel = train_kernel
        self.preload_test_data = preload_test_data
        self.test_kernel = test_kernel
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        
        self.res_logs=[]
        self.task_feat = {}
        self.task_k = {}

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)
            s_idxs = []
            for data in s_data:
                s_idxs.append(data.id[0].item())
            adjs = []
            for data in q_data:
                q_idxs = [data.id[0].item()]
                idxs = s_idxs + q_idxs
                #print(idxs)
                adj = self.train_kernel[task][np.ix_(idxs,idxs)]
                #print(adj)
                adj[np.isnan(adj)] = 0
                adjs.append(adj)
            kernel_adj = torch.tensor(np.array(adjs),dtype=torch.float32).cuda() 
            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)   

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0), 'kernel_adj': kernel_adj}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            '''
            t1 = time()
            s_idxs = []
            for data in s_data:
                s_idxs.append(data.id[0].item())
            adjs_list = []
            q_data_list = []
            for i in range(0, len(q_data), self.n_query):
                end_idx = max(i + self.n_query, len(q_data))
                batch_data = q_data[i:end_idx]
                adjs = []
                for data in batch_data:
                    q_idxs = [data.id[0].item()]
                    idxs = s_idxs + q_idxs
                    adj = self.test_kernel[task][np.ix_(idxs,idxs)]
                    adj[np.isnan(adj)] = 0
                    adjs.append(adj)
                kernel_adj = torch.tensor(np.array(adjs),dtype=torch.float32).cuda() 
                adjs_list.append(kernel_adj)
                batch_data = self.loader_to_samples(batch_data)
                q_data_list.append(batch_data)
            q_data_adapt_list = []
            adjs_adapt_list = []
            for i in range(0, len(q_data_adapt), self.n_query):
                end_idx = max(i + self.n_query, len(q_data))
                batch_data = q_data_adapt[i:end_idx]
                adjs = []
                for data in batch_data:
                    q_idxs = [data.id[0].item()]
                    idxs = s_idxs + q_idxs
                    adj = self.test_kernel[task][np.ix_(idxs,idxs)]
                    adj[np.isnan(adj)] = 0
                    adjs.append(adj)
                kernel_adj = torch.tensor(np.array(adjs),dtype=torch.float32).cuda() 
                adjs_adapt_list.append(kernel_adj)
                batch_data = self.loader_to_samples(batch_data)
                q_data_adapt_list.append(batch_data)
            '''
            s_data = self.loader_to_samples(s_data)
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query*4, shuffle=True, num_workers=0)
            #q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)
            #print("time cost:", (time() - t1)/60)
            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt, 'kernel_adj': self.test_kernel[task]}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader, 'kernel_adj': self.test_kernel[task]}

        return adapt_data, eval_data

    def get_prediction(self, model, data, train=True, task_id=None):
        if train:
            s_logits, q_logits, adj, node_emb, s_logits_augs, q_logits_augs, task_feat, s_emb_augs = model(data['s_data'], data['q_data'], data['s_label'], kernel_simi=data['kernel_adj'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb, 's_logits_augs': s_logits_augs, 'q_logits_augs': q_logits_augs, 'task_feat': task_feat, 's_emb_augs': s_emb_augs}

        else:
            s_logits, logits,labels,adj_list,sup_labels, s_logits_augs = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'], kernel_simi=data['kernel_adj'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels, 's_logits_augs': s_logits_augs}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel = lambda x: x[0]== 'adapt_relation'
        fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
        fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
        fnode2 = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1] and ('network' not in x[2])
        fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not frel(x)
        elif adapt_weight==2:
            flag=lambda x: not (fenc(x) or frel(x))
        elif adapt_weight==3:
            flag=lambda x: not (fenc(x) or fedge(x))
        elif adapt_weight==4:
            flag=lambda x: not (fenc(x) or fnode(x))
        elif adapt_weight==5:
            flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
        elif adapt_weight==6:
            flag=lambda x: not (fenc(x) or fclf(x))
        elif adapt_weight==7:
            flag=lambda x: not (fenc(x) or fnode2(x) or fedge(x))
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query
        if not train:
            losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_test*n_query,2), batch_data['s_label'].repeat(n_query))
            losses_aug = 0
            for s_logits_aug in pred_dict['s_logits_augs']:
                losses_aug += self.criterion(s_logits_aug.reshape(2*n_support_train*n_query,2), batch_data['s_label'].repeat(n_query))
            losses_aug /= len(pred_dict['s_logits_augs'])
            #losses_consi = consis_loss(pred_dict['s_logits_augs']) + consis_loss(pred_dict['s_logits'])
            losses_consi = consis_loss(pred_dict['s_logits_augs'])
            #losses_adapt += losses_aug + losses_consi
            losses_adapt += self.args.consi * losses_consi
        else:
            if flag:
                losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_train*n_query,2), batch_data['s_label'].repeat(n_query))
                losses_aug = 0
                for s_logits_aug in pred_dict['s_logits_augs']:
                    losses_aug += self.criterion(s_logits_aug.reshape(2*n_support_train*n_query,2), batch_data['s_label'].repeat(n_query))
                losses_aug /= len(pred_dict['s_logits_augs'])
                losses_consi = consis_loss(pred_dict['s_logits_augs'])
            else:
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
                losses_aug = 0
                for q_logits_aug in pred_dict['q_logits_augs']:
                    losses_aug += self.criterion(q_logits_aug, batch_data['q_label'])
                losses_aug /= len(pred_dict['q_logits_augs'])
                #losses_consi = consis_loss(pred_dict['s_logits_augs']) + consis_loss(pred_dict['q_logits_augs'])
                #losses_consi = consis_loss(pred_dict['s_logits_augs']) + consis_loss(pred_dict['s_logits'])
                losses_consi = consis_loss(pred_dict['s_logits_augs']) + consis_loss(pred_dict['s_emb_augs'])
                #losses_consi = consis_loss(pred_dict['s_emb_augs'])
            #losses_adapt += losses_aug + losses_consi
            losses_adapt += self.args.consi * losses_consi
        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            print(pred_dict['s_logits'])
            losses_adapt = torch.zeros_like(losses_adapt)
        if self.args.reg_adj > 0:
            n_support = batch_data['s_label'].size(0)
            adj = pred_dict['adj'][-1]
            if train:
                if flag:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    n_d = n_query * n_support
                    label_edge = model.label2edge(s_label).reshape((n_d, -1))
                    pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
                else:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = torch.cat((s_label, q_label), 1)
                    label_edge = model.label2edge(total_label)[:,:,-1,:-1]
                    pred_edge = adj[:,:,-1,:-1]
            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support
                label_edge = model.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
            adj_loss_val = F.mse_loss(pred_edge, label_edge)
            if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
                adj_loss_val = torch.zeros_like(adj_loss_val)

            losses_adapt += self.args.reg_adj * adj_loss_val

        return losses_adapt
    
    def set_aug_rules(self, task_id, train):
        if train:
            task = self.train_tasks[task_id]
            dataset = self.preload_train_data[task]
            smiles = dataset.smiles_list
            self.model.module.smiles_list = smiles
            self.model.module.rules = self.rules        
            self.model.module.rule_indicator = self.train_rules[task]
        else:
            task = self.test_tasks[task_id]
            dataset = self.preload_test_data[task]
            smiles = dataset.smiles_list
            self.model.module.smiles_list = smiles
            self.model.module.rules = self.rules        
            self.model.module.rule_indicator = self.test_rules[task]
        
    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db
        
        for k in range(self.update_step):
            losses_eval = []
            #task_feats = []
            task_weights = {}
            for task_id in task_id_list:
                #print("train:", task_id)
                #gpu_tracker.track() 
                self.set_aug_rules(task_id, True)
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                self.mlp.eval()
                adaptable_weights = self.get_adaptable_weights(model)
                
                pred_adapt = self.get_prediction(model, train_data, train=True)
                task_feat = pred_adapt["task_feat"]
                if task_id not in self.task_feat:
                    self.task_feat[task_id] = task_feat.detach().cpu()
                    self.task_k[task_id] = 1
                else:
                    self.task_k[task_id] += 1
                    k = self.task_k[task_id]
                    task_feat0 = self.task_feat[task_id]
                    self.task_feat[task_id] = task_feat0 * (k - 1) / k + task_feat.detach().cpu() / k
                #torch.cuda.empty_cache()
                task_feats = []
                for idx in self.task_feat:
                    task_feats.append(self.task_feat[idx].cuda())
                #gpu_tracker.track() 
                mask_w, mask_b, weights = self.mlp(task_feat, torch.stack(task_feats))
                del task_feat
                del task_feats
                del pred_adapt
                task_weights[task_id] = weights
                #gpu_tracker.track() 
                model.adapt_relation.fc2.weight = nn.Parameter(model.adapt_relation.fc2.weight * mask_w.reshape(2, self.args.map_dim))
                model.adapt_relation.fc2.bias = nn.Parameter(model.adapt_relation.fc2.bias * mask_b)
                
                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    #task_feat = pred_adapt["task_feat"]
                    #task_weight = self.attention2(task_feat)[0]
                    #loss_adapt = task_weight * self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
                    model.adapt(loss_adapt, adaptable_weights = adaptable_weights)

                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)
                #task_feat = pred_eval["task_feat"]
                #self.task_feat[task_id] = task_feat

                losses_eval.append(loss_eval)
                torch.cuda.empty_cache()
                #task_feats.append(task_feat)

            #print(losses_eval)
            #print(task_feats)
            losses_eval = torch.stack(losses_eval)
            #task_feats = torch.stack(task_feats)
            #task_weights = self.attention(task_feats)
            #losses_eval = torch.sum(losses_eval * task_weights)

            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)
            self.mlp.train()
            self.optimizer.zero_grad()
            self.optimizer_mlp.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), 1)
            self.optimizer.step()
            self.optimizer_mlp.step()
            print('Train Epoch:'+ str(self.train_epoch) + ', train update step:'+str(k)+ ', loss_eval:'+str(losses_eval.item()))
            

        return self.model.module
    
    def train_step_old(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db
        #global train_task
        for k in range(self.update_step):
            #global train_task
            def train_task(x):
                train_data, _ = data_batches[x]
                loss_eval = 0
                
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)
                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    task_feat = pred_adapt["task_feat"]
                    #task_weight = self.attention2(task_feat)[0]
                    #loss_adapt = task_weight * self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
                    model.adapt(loss_adapt, adaptable_weights = adaptable_weights)
                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)
                task_feat = pred_eval["task_feat"]
                
                return loss_eval
            
            losses_eval = self.pool.map(train_task, task_id_list)
            self.pool.close()
            self.pool.join()
            losses_eval = torch.stack(losses_eval)
            #task_feats = torch.stack(task_feats)
            #task_weights = self.attention(task_feats)
            #losses_eval = torch.sum(losses_eval * task_weights)

            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)
            #self.attention2.train()
            self.optimizer.zero_grad()
            #self.optimizer_attn2.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            #self.optimizer_attn2.step()

            print('Train Epoch:'+ str(self.train_epoch) + ', train update step:'+str(k)+ ', loss_eval:'+str(losses_eval.item()))

        return self.model.module
    
    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        #global test_task
        task_weights = {}
        for task_id in range(len(self.test_tasks)):
            self.set_aug_rules(task_id, False)
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()
            if self.update_step_test>0:
                model.train()
                self.mlp.eval()
                for i, batch in enumerate(adapt_data['data_loader']):
                    torch.cuda.empty_cache()
                    batch = batch.to(self.device)
                    #rint(batch.shape, adapt_data['kernel_adj'][i].shape)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                        'q_data': batch, 'q_label': None, 'kernel_adj': adapt_data['kernel_adj']}

                    adaptable_weights = self.get_adaptable_weights(model)
                    task_feats = []
                    for idx in self.task_feat:
                        task_feats.append(self.task_feat[idx].cuda())
                    pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    task_feat = pred_adapt["task_feat"]
                    mask_w, mask_b, weights = self.mlp(task_feat, torch.stack(task_feats))
                    del task_feat
                    del task_feats, pred_adapt
                    model.adapt_relation.fc2.weight = nn.Parameter(model.adapt_relation.fc2.weight * mask_w.reshape(2, self.args.map_dim))
                    model.adapt_relation.fc2.bias = nn.Parameter(model.adapt_relation.fc2.bias * mask_b)
                    
                    pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    #task_feat = pred_adapt["task_feat"]
                    #task_weight = self.attention2(task_feat)[0]
                    #loss_adapt = task_weight * self.get_loss(model, cur_adapt_data, pred_adapt, train=False)
                    loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

                    model.adapt(loss_adapt, adaptable_weights=adaptable_weights)

                    if i>= self.update_step_test-1:
                        break

            model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)
                y_score = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
                y_true = pred_eval['labels']
                if self.args.eval_support:
                    y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                    y_s_true = eval_data['s_label']
                    y_score=torch.cat([y_score, y_s_score])
                    y_true=torch.cat([y_true, y_s_true])
                auc = auroc(y_score,y_true,pos_label=1).item()
            auc_scores.append(auc)
            task_weights[task_id] = weights
            
            del weights
            print('Test Epoch:'+ str(self.train_epoch) + ', test for task:'+ str(task_id) + ', AUC:'+ str(round(auc, 4)))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])
            
            
        #auc_scores = self.pool.map(test_task, list(range(len(self.test_tasks))))
        #self.pool.close()
        #self.pool.join()
        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        #if avg_auc > 0.67:
            #w = np.array([task_weights[task] for task in task_weights])
            #np.save("weights_" + str(self.train_epoch) + ".npy", w)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:'+str(self.train_epoch)+ ', AUC_Mid:'+str(round(mid_auc, 4))+', AUC_Avg: '+str(round(avg_auc, 4))+
              ', Best_Avg_AUC: '+str(round(self.best_auc, 4)))
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
