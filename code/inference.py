import torch
from tqdm import tqdm, trange
import os
import numpy as np
import shutil
import math
import sys
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from utils import model_util
from dataloaders import load_data
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_emb(model, d, graph_batcher=None, mol2text=0):

    with torch.no_grad():

        cid = np.array([d['cid']])
        text_mask = torch.Tensor(d['input']['text']['attention_mask']).bool().reshape(1,-1).to(device)
        text = torch.Tensor(d['input']['text']['input_ids']).long().reshape(1,-1).to(device)
        molecule = torch.Tensor(d['input']['molecule']['mol2vec']).float().reshape(1,-1).to(device)
        
        if graph_batcher:
            graph_batch = graph_batcher([d['input']['molecule']['cid']]).to(device)
            graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))
            outputs = model(text, graph_batch, text_mask)
        else:
            outputs = model(text, molecule, text_mask)
        
        text_emb = outputs[0]
        chem_emb = outputs[1]
        chem_emb = chem_emb.cpu().numpy()
        text_emb = text_emb.cpu().numpy()

    if mol2text == 1:
        return cid, text_emb, chem_emb
    else:
        return cid, chem_emb, text_emb


def extract_embeddings(args, model, logger, gd, graph_batchers):
    cids_train = np.array([])
    cids_val = np.array([])
    cids_test = np.array([])
    chem_embeddings_train = np.array([])
    text_embeddings_train = np.array([])
    chem_embeddings_val = np.array([])
    text_embeddings_val = np.array([])
    chem_embeddings_test = np.array([])
    text_embeddings_test = np.array([])

    MODEL = args.model
    train_data, val_data, test_data = gd.generate_examples_train(), gd.generate_examples_val(), gd.generate_examples_test()
    graph_batcher_tr, graph_batcher_val, graph_batcher_test = graph_batchers
    for i, d in tqdm(enumerate(train_data)):
        if MODEL == "MLP":
            cid, chem_emb, text_emb = get_emb(model, d, mol2text=args.mol2text)
        else:
            cid, chem_emb, text_emb = get_emb(model, d, graph_batcher_tr, mol2text=args.mol2text)

        cids_train = np.concatenate((cids_train, cid)) if cids_train.size else cid
        chem_embeddings_train = np.concatenate((chem_embeddings_train, chem_emb)) if chem_embeddings_train.size else chem_emb
        text_embeddings_train = np.concatenate((text_embeddings_train, text_emb)) if text_embeddings_train.size else text_emb

        if (i+1) % 1000 == 0: logger.info("{} embeddings processed".format(i + 1))

    logger.info("Training Embeddings done:", cids_train.shape, chem_embeddings_train.shape)

    for d in tqdm(val_data):

        if MODEL == "MLP":
            cid, chem_emb, text_emb = get_emb(model, d, mol2text=args.mol2text)
        else:
            cid, chem_emb, text_emb = get_emb(model, d, graph_batcher_val, mol2text=args.mol2text)

        cids_val = np.concatenate((cids_val, cid)) if cids_val.size else cid
        chem_embeddings_val = np.concatenate((chem_embeddings_val, chem_emb)) if chem_embeddings_val.size else chem_emb
        text_embeddings_val = np.concatenate((text_embeddings_val, text_emb)) if text_embeddings_val.size else text_emb

    logger.info("Validation Embeddings done:{} {}".format(cids_val.shape, chem_embeddings_val.shape))

    for d in tqdm(test_data):
        
        if MODEL == "MLP":
            cid, chem_emb, text_emb = get_emb(model, d, mol2text=args.mol2text)
        else:
            cid, chem_emb, text_emb = get_emb(model, d, graph_batcher_test, mol2text=args.mol2text)

        cids_test = np.concatenate((cids_test, cid)) if cids_test.size else cid
        chem_embeddings_test = np.concatenate((chem_embeddings_test, chem_emb)) if chem_embeddings_test.size else chem_emb
        text_embeddings_test = np.concatenate((text_embeddings_test, text_emb)) if text_embeddings_test.size else text_emb

    logger.info("Test Embeddings done: {} {}".format(cids_test.shape, chem_embeddings_test.shape))
    
    emb_path = os.path.join(args.output_path, "embeddings/")
    np.save(os.path.join(emb_path, "cids_train.npy"), cids_train)
    np.save(os.path.join(emb_path, "cids_val.npy"), cids_val)
    np.save(os.path.join(emb_path, "cids_test.npy"), cids_test)
    np.save(os.path.join(emb_path, "chem_embeddings_train.npy"), chem_embeddings_train)
    np.save(os.path.join(emb_path, "chem_embeddings_val.npy"), chem_embeddings_val)
    np.save(os.path.join(emb_path, "chem_embeddings_test.npy"), chem_embeddings_test)
    np.save(os.path.join(emb_path, "text_embeddings_train.npy"), text_embeddings_train)
    np.save(os.path.join(emb_path, "text_embeddings_val.npy"), text_embeddings_val)
    np.save(os.path.join(emb_path, "text_embeddings_test.npy"), text_embeddings_test)

    logger.info("save embeddings to {}".format(emb_path))

    return (cids_train, cids_val, cids_test), (chem_embeddings_train, chem_embeddings_val, chem_embeddings_test), (text_embeddings_train, text_embeddings_val, text_embeddings_test)

def memory_efficient_similarity_matrix_custom(func, embedding1, embedding2, chunk_size = 1000):
    rows = embedding1.shape[0]
    
    num_chunks = int(np.ceil(rows / chunk_size))
    
    for i in range(num_chunks):
        end_chunk = (i+1)*(chunk_size) if (i+1)*(chunk_size) < rows else rows #account for smaller chunk at end...
        yield func(embedding1[i*chunk_size:end_chunk,:], embedding2)

def get_ranks(text_chem_cos, ranks_avg, offset, split= ""):
    ranks_tmp = []
    j = 0 #keep track of all loops
    for l, emb in enumerate(text_chem_cos):
        for k in range(emb.shape[0]):
            cid_locs = np.argsort(emb[k,:])[::-1]
            ranks = np.argsort(cid_locs) 
            
            ranks_avg[j,:] = ranks_avg[j,:] + ranks 
            
            rank = ranks[j+offset] + 1
            ranks_tmp.append(rank)
            

            j += 1
            if j % 1000 == 0: print(j, split+" processed")

    return np.array(ranks_tmp)

def test(args, model, logger, gd, graph_batchers):

    _, _, _, graph_batcher_tr, graph_batcher_val, graph_batcher_test, gd = load_data(args, bsz=150)
    graph_batchers = (graph_batcher_tr, graph_batcher_val, graph_batcher_test)

    logger.info("start testing")
    cids, chem_embeddings, text_embeddings = extract_embeddings(args, model, logger, gd, graph_batchers)
    #combine all splits:
    all_text_embbedings = np.concatenate(text_embeddings, axis = 0)
    all_mol_embeddings = np.concatenate(chem_embeddings, axis = 0)
    all_cids = np.concatenate(cids, axis = 0)
    n_train = len(cids[0])
    n_val = len(cids[1])
    n_test = len(cids[2])
    n = n_train + n_val + n_test
    offset_val = n_train
    offset_test = n_train + n_val

    text_chem_cos = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings[0], all_mol_embeddings)
    text_chem_cos_val = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings[1], all_mol_embeddings)
    text_chem_cos_test = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings[2], all_mol_embeddings)

    #Calculate Ranks:
    tr_avg_ranks = np.zeros((n_train, n))
    val_avg_ranks = np.zeros((n_val, n))
    test_avg_ranks = np.zeros((n_test, n))
    ranks_train = []
    ranks_val = []
    ranks_test = []

    ranks_train = get_ranks(text_chem_cos, tr_avg_ranks, offset=0, split="train")
    ranks_val = get_ranks(text_chem_cos_val, val_avg_ranks, offset=offset_val, split="val")
    ranks_test = get_ranks(text_chem_cos_test, test_avg_ranks, offset=offset_test, split="test")

    train_hits1 = np.mean(ranks_train <= 1)
    test_hits1 = np.mean(ranks_test <= 1)
    val_hits1 = np.mean(ranks_val <= 1)

    train_hits10 = np.mean(ranks_train <= 10)
    test_hits10 = np.mean(ranks_test <= 10)
    val_hits10 = np.mean(ranks_val <= 10)

    train_mrr = np.mean(1/ranks_train)
    test_mrr = np.mean(1/ranks_test)
    val_mrr = np.mean(1/ranks_val)

    train_mr = np.mean(ranks_train)
    test_mr = np.mean(ranks_test)
    val_mr = np.mean(ranks_val)
    
    if args.mol2text == 1:
        test_res = {"test_m2t/test_hits1": test_hits1, 
                    "test_m2t/test_hits10": test_hits10,
                    "test_m2t/test_mr": test_mr, 
                    "test_m2t/test_mrr": test_mrr,
                    "test_m2t/train_hits1": train_hits1, 
                    "test_m2t/train_hits10": train_hits10,
                    "test_m2t/train_mr": train_mr, 
                    "test_m2t/train_mrr": train_mrr,
                    "test_m2t/val_hits1": val_hits1, 
                    "test_m2t/val_hits10": val_hits10,
                    "test_m2t/val_mr": val_mr, 
                    "test_m2t/val_mrr": val_mrr}
    else:
        test_res = {"test/test_hits1": test_hits1, "test/test_hits10": test_hits10,
                    "test/test_mr": test_mr, "test/test_mrr": test_mrr,
                    "test/train_hits1": train_hits1, "test/train_hits10": train_hits10,
                    "test/train_mr": train_mr, "test/train_mrr": train_mrr,
                    "test/val_hits1": val_hits1, "test/val_hits10": val_hits10,
                    "test/val_mr": val_mr, "test/val_mrr": val_mrr}
    print(test_res)
    return test_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CMTMR')
    parser.add_argument('--data', type=str, default="/home/text2mol")
    parser.add_argument('--home-dir', type=str, default="/home/text2mol")
    parser.add_argument('--model', type=str, default='MLP', nargs='?',
                    help="model type from 'MLP, 'GCN', 'Attention'")
    parser.add_argument('--model-name', type=str, default='MLP')
    parser.add_argument('--mol_trunc_length', type=int, nargs='?', default=512)
    parser.add_argument('--text_trunc_length', type=int, nargs='?', default=256)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output-path', type=str, default="/home/output/test")
    parser.add_argument('--ckpt-path', type=str, default="")
    parser.add_argument('--ninp', type=int, default=768)
    parser.add_argument('--nhid', type=int, default=600)
    parser.add_argument('--nout', type=int, default=300)
    parser.add_argument('--num-node-features', type=int, default=300)
    parser.add_argument('--graph-hidden-channels', type=int, default=600)
    parser.add_argument('--kl-type', type=str, default="inference")
    parser.add_argument('--modal-embedding', type=int, default=0)
    parser.add_argument('--interlingua-length', type=int, default=16)
    parser.add_argument('--interlingua-layer-num', type=int, default=3)
    parser.add_argument('--contra-weight', type=float, default=1.0)
    parser.add_argument('--ma-type', type=str, default="query")
    parser.add_argument('--query-update', type=str, default="cl")
    parser.add_argument('--gan-weight', type=float, default=2e-3)
    parser.add_argument('--mol2text', type=int, default=0)
    args = parser.parse_args()  

    # load model
    model = model_util.load_model(args)
    model_util.load_ckpt(args.ckpt_path, model)
    model = model.cuda()
    model.eval()
    logger = logging.getLogger("test")
    _, _, _, graph_batcher_tr, graph_batcher_val, graph_batcher_test, gd = load_data(args)
    graph_batchers = (graph_batcher_tr, graph_batcher_val, graph_batcher_test)
    print(test(args, model, logger, gd, graph_batchers))