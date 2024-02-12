import pandas as pd
import os
import numpy as np
from pathlm.data_mappers.mapper_torchkge import dataset_preprocessing
from torchkge import KnowledgeGraph
import torch
from collections import defaultdict

def get_weight_dir(method_name: str, dataset: str):
    weight_dir = os.path.join("pathlm/models/kge_rec/",method_name,"weight_dir")
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)
    return weight_dir

def get_weight_ckpt_dir(method_name: str,dataset: str):
    weight_dir_ckpt = os.path.join("pathlm/models/kge_rec/",method_name,"weight_dir_ckpt")
    if not os.path.isdir(weight_dir_ckpt):
        os.makedirs(weight_dir_ckpt)
    return weight_dir_ckpt

def get_test_uids(dataset):
    preprocessed_torchkge=f"data/{dataset}/preprocessed/torchkge"
    test_path = f"{preprocessed_torchkge}/triplets_test.txt"
    kg_df_test = pd.read_csv(test_path, sep="\t")
    kg_df_test.rename(columns={"0":"from","1":"to","2":"rel"},inplace=True)
    uids=np.unique(kg_df_test['from'])
    return uids

def get_log_dir(method_name: str):
    log_dir = os.path.join("pathlm/models/kge_rec/",method_name,"log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir

def load_kg(dataset): 
    preprocessed_torchkge=f"data/{dataset}/preprocessed/torchkge"
    dataset_preprocessing(dataset)
    e_df_withUsers=pd.read_csv(f"{preprocessed_torchkge}/e_map_with_users.txt", sep="\t")
    kg_df = pd.read_csv(f"{preprocessed_torchkge}/kg_final_updated.txt", sep="\t")
    kg_df.rename(columns={"entity_head":"from","entity_tail":"to","relation":"rel"},inplace=True)
    kg_train=KnowledgeGraph(df=kg_df,ent2ix=dict(zip(e_df_withUsers['eid'],e_df_withUsers['eid'])))
    return kg_train

def get_preprocessed_torchkge_path(dataset):
    preprocessed_torchkge=f"data/{dataset}/preprocessed/torchkge"
    return preprocessed_torchkge

def get_users_positives(path): 
    users_positives=dict()
    with open(f"{path}/triplets_train_valid.txt","r") as f:
        for i,row in enumerate(f):
            if i==0:
                continue
            uid,pid,_=row.split("\t")
            uid=int(uid)
            pid=int(pid)
            if uid in users_positives:
                users_positives[uid].append(pid)
            else:
                users_positives[uid]=[pid]
    return users_positives
 
def remap_topks2datasetid(args,topks):
    """load entities and user_mapping"""
    e_new_df=pd.read_csv(f"{args.preprocessed_torchkge}/e_map_with_users.txt",sep="\t")
    user_mapping=pd.read_csv(f"{args.preprocessed_torchkge}/user_mapping.txt",sep="\t")

    """create the correct mapping"""
    torchkgeid2datasetuid={eid:entity for eid, entity in zip(e_new_df['eid'],e_new_df['entity'])} # mapping uid
    datasetid2useruid={int(datasetid):int(userid) for datasetid, userid in zip(user_mapping['rating_id'],user_mapping['new_id'])}

    """Mapping users to correct uid"""
    topks={int(datasetid2useruid[int(torchkgeid2datasetuid[key])]):values for key, values in topks.items()}
    
    return topks

"""Utils functions for TuckER Model"""
def get_data_idxs(data,entity_idxs, relation_idxs):
        data_idxs = [(entity_idxs[data[i][0]], relation_idxs[data[i][1]], \
                      entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
def get_er_vocab(data):
    er_vocab = defaultdict(list)# se non ha la chiave che gli chiedi, crea {chiave:[]}
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    return er_vocab

def get_batch(d,batch_size,er_vocab, er_vocab_pairs, idx,use_cuda=True):
    batch = er_vocab_pairs[idx:idx+batch_size]
    targets = np.zeros((len(batch), len(d.entities)))
    for idx, pair in enumerate(batch):
        targets[idx, er_vocab[pair]] = 1.
    targets = torch.FloatTensor(targets)
    if use_cuda:
        targets = targets.cuda()
    return np.array(batch), targets
