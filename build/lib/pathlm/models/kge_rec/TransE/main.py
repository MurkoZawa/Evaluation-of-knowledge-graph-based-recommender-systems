"""
TRAIN + EVALUATE
"""
method_name="TransE"
import logging
import torch
import numpy as np
import pandas as pd
import os
from time import time
import random
from tqdm.autonotebook import tqdm
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from pathlm.utils import SEED
#from pathlm.models.embeddings.kge_utils import get_log_dir
from pathlm.models.model_utils import EarlyStopping, logging_metrics
from pathlm.models.traditional.log_helper import logging_config, create_log_id
from pathlm.datasets.data_utils import get_set
from pathlm.evaluation.eval_metrics import evaluate_rec_quality
from transe import MarginLoss
from transe import TransE
from pathlm.models.kge_rec.TransE.parser_transe import parse_args
from pathlm.data_mappers.mapper_torchkge import get_watched_relation_idx

"""Utils"""
from pathlm.models.kge_rec.utils import get_test_uids, get_log_dir,load_kg,get_users_positives,remap_topks2datasetid


def initialize_model(kg_train,b_size,emb_dim,weight_decay,margin,lr,use_cuda):
    """Define Model"""
    model = TransE(kg_train.n_ent, kg_train.n_rel, emb_dim)
    """Loss"""
    criterion = MarginLoss(margin=margin)
    """Define the torch optimizer to be used"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    """Define negative sampler"""
    sampler = BernoulliNegativeSampler(kg_train)
    """Define Dataloader"""
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=use_cuda)
    return model,criterion, optimizer, sampler,dataloader


def train_epoch(model,sampler,optimizer,criterion,dataloader,epoch,args):
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r,n_neg=5)

        optimizer.zero_grad()

        """forward + backward + optimize"""
        pos, neg = model(h, t, n_h, n_t, r)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i % args.print_every) == 0:
            # per debugging metti logging.warning cosi appare in stdout
            logging.info(f'KG Training: Epoch {epoch:04d} Iter {i:04d} / {args.epoch:04d} | Iter Loss {running_loss:.4f}')

    return running_loss

def print_training_info(epoch, train_time, loss):
    logging.info(f"Epoch {epoch} [{train_time:.1f}s]: Loss: {loss:.5f}")


def evaluate_model(model,args):
    """Normalize parameters after each training"""
    model.normalize_parameters()
    
    """Get Watched Relation"""
    WATCHED=get_watched_relation_idx(args.dataset)
    
    """Load Pids identifiers"""
    pids_identifiers=np.load(f"{args.preprocessed_torchkge}/pids_identifiers_new.npy")
    kg_train=load_kg(args.dataset)
    
    """Get kg test uids"""
    uids=get_test_uids(args.dataset)
    
    """Get users_positives, pids the user has already interacted with"""
    users_positives=get_users_positives(args.preprocessed_torchkge)

    """Load Embeddings"""
    entities_emb, relations_emb=model.get_embeddings()
    
    """Learning To Rank"""
    top_k_recommendations={}
    for uid in uids:
        remapped_uid=kg_train.ent2ix[uid]
        user_emb=entities_emb[remapped_uid]
        products_emb=entities_emb[[kg_train.ent2ix[pid] for pid in pids_identifiers]]
        user_rel_emb=user_emb+relations_emb[kg_train.rel2ix[WATCHED]]
        dot_prod=np.dot(user_rel_emb.cpu(),products_emb.T.cpu())# score di 1 utente per tutti i prodotti
        users_positives_mask=[kg_train.ent2ix[pid] for pid in users_positives[uid]]
        dot_prod[users_positives_mask]=float('-inf')
        indexes=np.argsort(dot_prod)[::-1]
        top_k_recommendations[uid]=indexes[:args.K]

    """Remap uid of top_k_recommendations to dataset id"""
    top_k_recommendations=remap_topks2datasetid(args,top_k_recommendations)
    test_labels=get_set(args.dataset,'test')
    _,avg_rec_quality_metrics=evaluate_rec_quality(args.dataset, top_k_recommendations, test_labels, args.K,method_name=method_name)

    return avg_rec_quality_metrics,top_k_recommendations



def train(args):
    """Set random seeds for reproducibility"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    """Setup logging"""
    log_dir=get_log_dir(method_name)
    log_save_id = create_log_id(log_dir)
    logging_config(folder=log_dir, name=f'log{log_save_id}', no_console=True)
    logging.info(args)

    """Setup device (GPU/CPU)"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.device = device
    
    #train_cores = multiprocessing.cpu_count()

    """Load kg_train and initialize model"""
    kg_train=load_kg(args.dataset)
    model,criterion,optimizer,sampler, dataloader=initialize_model(kg_train,args.batch_size,args.embed_size,args.weight_decay,args.margin,args.lr,args.use_cuda)

    """Move everything to MPS or CUDA or CPU if available"""
    model.to(args.device)
    criterion.to(args.device)

    """Training loop"""
    logging.info(model)
    early_stopping = EarlyStopping(patience=15, verbose=True)

    iterator = tqdm(range(args.epoch), unit='epoch')
    for epoch in iterator:
        t1 = time()
        """Phase 1: CF training"""
        running_loss=train_epoch(model,sampler,optimizer,criterion, dataloader,epoch,args)
        print_training_info(epoch, time() - t1,running_loss)
        assert np.isnan(running_loss) == False

        """
        Phase 2: Test
        Testing and performance logging
        """
        test_metrics, topks=evaluate_model(model,args)
        logging_metrics(epoch, test_metrics, [str(args.K)])

        ndcg_value = test_metrics['ndcg']
        early_stopping(ndcg_value)

        if early_stopping.early_stop:
            logging.info('Early stopping triggered. Stopping training.')
            break

        """Optional: Save model and metrics at each epoch or at specific intervals"""
        if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
            torch.save(model.state_dict(), os.path.join(args.weight_dir_ckpt, f'{method_name}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
        
        
        iterator.set_description('Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))
    
    """Final model save and cleanup"""
    torch.save(model.state_dict(), os.path.join(args.weight_dir, f'{method_name}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
    logging.info(f'Best evaluation results at epoch {early_stopping.best_epoch} with NDCG: {early_stopping.best_score:.4f}')



if __name__ == '__main__':
    args = parse_args()
    train(args)
