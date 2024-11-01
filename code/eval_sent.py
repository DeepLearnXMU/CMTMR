import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm, trange
from torch_geometric.data import Dataset
import torch_geometric
from torch.utils.data import RandomSampler
import os
from utils.model_util import save_model, load_model, load_ckpt
from transformers import BertTokenizerFast, BertModel

class GINMatchDataset(Dataset):
    def __init__(self, args):
        super(GINMatchDataset, self).__init__()
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir('/home/PCdes_sent/graph-data/processed')
        self.text_name_list = os.listdir('/home/PCdes_sent/text')
        self.tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")

    def __len__(self):
        return len(self.text_name_list)

    def __getitem__(self, index):
        text_name = self.text_name_list[index]
        text_path = os.path.join('/home/PCdes_sent/text', text_name)
        cid = int(text_name.split(".")[0])
        graph_name = "data_{}.pt".format(cid)
        graph_path = os.path.join('/home/PCdes_sent/graph-data/processed', graph_name)
        data_graph = torch.load(graph_path)
        data_aug = data_graph
        text_path = os.path.join('/home/PCdes_sent/text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 500:
                break
        
        text = mask = None

        sts = text_list[0].split('.')
        remove_list = []
        for st in (sts):
            if len(st.split(' ')) < 5: 
                remove_list.append(st)
        remove_list = sorted(remove_list, key=len, reverse=False)
        for r in remove_list:
            if len(sts) > 1:
                sts.remove(r)
        text_index = random.randint(0, len(sts)-1)
        text, mask = self.tokenizer_text(sts[text_index])

        return data_aug, text.squeeze(0), mask.squeeze(0)#, index

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   max_length=256,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


class GINSentDataset(Dataset):
    def __init__(self, args):
        super(GINSentDataset, self).__init__()
        self.text_max_len = 256
        self.text_name_list = os.listdir('/home/PCdes_sent/text')
        self.tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
        
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        #self.cor.append(cnt)
        print("gin sent dataset len:", len(self.text_name_list))
        for text_name in self.text_name_list:
            text_path = os.path.join('/home/PCdes_sent/text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        print("gin sent dataset len(cor): ", len(self.cor), self.cor[0], self.cor[-1])
        np.save('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   max_length=256,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


def prepare_model_and_optimizer(args, device):
    model = load_model(args)
    load_ckpt(resume_path=args.init_checkpoint, model=model)
    model.to(device)
    return model, None

def Eval(model, dataloader, device, args):
    
    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
        graph_rep_total = None
        text_rep_total = None
        for batch in tqdm(dataloader):
            aug, text, mask = batch
            aug.to(device)
            text = text.cuda()
            mask = mask.cuda()
            mask = mask.bool()

            outputs = model(text, aug, mask)
            text_rep, graph_rep = outputs[0], outputs[1]

            scores1 = torch.cosine_similarity(graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1==torch.arange(argm1.shape[0]).cuda()).int()).item()
            acc2 += sum((argm2==torch.arange(argm2.shape[0]).cuda()).int()).item()

            allcnt += argm1.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/graph_rep.npy', graph_rep_total.cpu())
    np.save('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/text_rep.npy', text_rep_total.cpu())

    return acc1/allcnt, acc2/allcnt

# get every sentence's rep
def CalSent(model, dataloader, device, args): 
    model.eval()
    with torch.no_grad():
        text_rep_total = None
        for batch in tqdm(dataloader):
            text, mask = batch
            text = text.cuda()
            mask = mask.cuda()
            mask = mask.bool()
            text_rep = model.get_text_rep(text, mask)

            if text_rep_total is None:
                text_rep_total = text_rep
            else:
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/text_rep.npy', text_rep_total.cpu())

def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    model, _ = prepare_model_and_optimizer(args, device)

    TestSet = GINMatchDataset(args)
    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=1, pin_memory=True, drop_last=False)#True
    SentSet = GINSentDataset(args)
    sent_dataloader = torch_geometric.loader.DataLoader(SentSet, shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=1, pin_memory=True, drop_last=False)#True
    global_step = 0
    tag = True
    best_acc = 0

    acc1, acc2 = Eval(model, test_dataloader, device, args)
    print('Test Acc1:', acc1)
    print('Test Acc2:', acc2)
    graph_rep = torch.from_numpy(np.load('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/graph_rep.npy'))
    
    CalSent(model, sent_dataloader, device, args)
    graph_rep = torch.from_numpy(np.load('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/graph_rep.npy'))
    text_rep = torch.from_numpy(np.load('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/text_rep.npy'))
    cor = np.load('/home/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/cor.npy')
    
    graph_len = graph_rep.shape[0]
    text_len = text_rep.shape[0]

    score1 = torch.zeros(graph_len, graph_len)
    score2 = torch.zeros(graph_len, graph_len)

    for i in trange(graph_len):
        score = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
        for j in range(graph_len):
            total = 0
            for k in range(cor[j], cor[j+1]):
                total+=(score[k]/(cor[j+1]-cor[j]))
            score1[i,j] = total
            #score1[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
    rec1 = []
    for i in trange(graph_len):
        a,idx = torch.sort(score1[:,i])
        for j in range(graph_len):
            if idx[-1-j]==i:
                rec1.append(j)
                break
    print(f'Rec@20 1: {sum( (np.array(rec1)<20).astype(int) ) / graph_len}')

    score_tmp = torch.zeros(text_len, graph_len)
    for i in range(text_len):
        score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
    score_tmp = torch.t(score_tmp)

    for i in range(graph_len):
        for j in range(graph_len):
            total = 0
            for k in range(cor[j], cor[j+1]):
                total+=(score_tmp[i][k]/(cor[j+1]-cor[j]))
            score2[i,j] = total
            #score2[i,j] = sum(score_tmp[i][cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
    score2 = torch.t(score2)

    rec2 = []
    for i in range(graph_len):
        a,idx = torch.sort(score2[:,i])
        for j in range(graph_len):
            if idx[-1-j]==i:
                rec2.append(j)
                break
    print(f'Rec@20 2: {sum( (np.array(rec2)<20).astype(int) ) / graph_len}')

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default="/home/CMTMR/output/finetune_sent_q32sota_pcdes/models/checkpoint_last.pt", type=str,)
    parser.add_argument("--data_type", default=1, type=int) # 0-para, 1-sent
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--seed", default=73, type=int,)
    parser.add_argument("--text_max_len", default=256, type=int,)
    parser.add_argument('--model', type=str, default='GCN_GAN_QUERY')
    parser.add_argument('--encoder-embed-dim', type=int, default=300) # nout
    parser.add_argument('--encoder-attention-heads', type=int, default=4)
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--encoder-normalize-before', type=bool, default=True)
    parser.add_argument('--encoder-ffn-embed-dim', type=int, default=1200)
    parser.add_argument('--ninp', type=int, default=768)
    parser.add_argument('--nhid', type=int, default=600)
    parser.add_argument('--nout', type=int, default=300)
    parser.add_argument('--num-node-features', type=int, default=300)
    parser.add_argument('--graph-hidden-channels', type=int, default=600)
    parser.add_argument('--modal-embedding', type=int, default=0)
    parser.add_argument('--interlingua-length', type=int, default=32)
    parser.add_argument('--interlingua-layer-num', type=int, default=3)
    parser.add_argument('--ma-type', type=str, default="query")
    parser.add_argument('--query-update', type=str, default="cl")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())