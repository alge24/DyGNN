import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from model_recurrent import DyGNN
from datasets import Temporal_Dataset
import argparse
from scipy.stats import rankdata
import numpy as np
import random
import os
import time

def get_args():
    parser = argparse.ArgumentParser(description = 'Show description')
    parser.add_argument('-data', '--dataset', type = str,
                        help = 'which dataset to run', default = 'uci')
    parser.add_argument('-b', '--batch_size', type= int,
                        help = 'batch_size', default = 200)
    parser.add_argument('-l', '--learning_rate', type = float,
                        help = 'learning_rate', default = 0.001)
    parser.add_argument('-nn', '--num_negative', type = int,
                        help = 'num_negative', default = 5)
    parser.add_argument('-tr', '--train_ratio', type = float,
                        help = 'train_ratio', default = 0.8)
    parser.add_argument('-vr', '--valid_ratio', type = float,
                        help = 'valid_ratio', default = 0.01)
    parser.add_argument('-act', '--act', type = str,
                        help = 'act function', default = 'tanh')
    parser.add_argument('-trans', '--transfer', type = int,
                        help = 'transfer to head, tail representations', default = 1)
    parser.add_argument('-dp' , '--drop_p', type = float,
                        help = 'dropout_rate', default = 0)
    parser.add_argument('-ip', '--if_propagation', type = int,
                        help = 'if_propagation', default=1)
    parser.add_argument('-ia', '--is_att', type = int,
                        help = 'use attention or not', default=1)
    parser.add_argument('-w', '--w', type = float,
                        help = 'w for decayer', default = 2)
    parser.add_argument('-s', '--seed', type = int,
                        help = 'random seed', default = 0)
    parser.add_argument('-rp', '--reset_rep', type = int,
                        help = 'whether reset rep', default = 1)
    parser.add_argument('-dc', '--decay_method', type = str,
                        help = 'decay_method', default = 'log')
    parser.add_argument('-nor', '--nor', type = int ,
                        help = 'normalize or not', default = 0)
    parser.add_argument('-iu', '--if_updated', type = int,
                        help = 'use updated representation in loss', default = 0)
    parser.add_argument('-wd', '--weight_decay', type = float,
                        help = 'weight decay', default = 0.001)
    parser.add_argument('-nt', '--if_no_time', type = int,
                        help = 'if no time interval information', default = 0)
    parser.add_argument('-th', '--threhold', type = float,
                        help = 'the threhold to filter the neighbors, if None, do not filter', default = None)
    parser.add_argument('-2hop', '--second_order', type = int,
                        help = 'whether to use 2-hop prop', default = 0)
    args = parser.parse_args()
    return args


def link_prediction(data, reps):
    head_list = list(data[:,0])
    tail_list = list(data[:,1])
    head_reps = reps[head_list,:]
    tail_reps = reps[tail_list,:]

def get_loss(data, head_reps, tail_reps,device):

    head_list = list(data[:,0])
    tail_list = list(data[:,1])


    head_tensors = head_reps(torch.LongTensor(head_list).to(device))
    tail_tensors = tail_reps(torch.LongTensor(tail_list).to(device))
    scores = torch.bmm(head_tensors.view(len(head_list),1,head_tensors.size()[1]),tail_tensors.view(len(head_list),head_tensors.size()[1],1)).view(len(head_list))
    labels = torch.FloatTensor([1]*len(head_list)).to(device)
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    loss = bce_with_logits_loss(scores,labels)
    return loss

def rank(node, true_candidate, node2candidate, node_reps, candidate_reps, device, pri = False):
    node_tensor = node_reps(torch.LongTensor([node]).to(device)).view(-1,1)
    candidates = list(node2candidate[node])

    candidates.append(true_candidate)

    length = len(candidates)

    candidate_tensors = candidate_reps(torch.LongTensor(candidates).to(device))

    scores = torch.mm(candidate_tensors, node_tensor)
    negative_scores_numpy = -scores.view(1,-1).to('cpu').numpy()
    rank = rankdata(negative_scores_numpy)[-1]

    if pri:
        print(node , true_candidate)
        print(scores.view(-1))
        print(rank, 'out of',length)

    return rank, length



def get_previous_links(data):
    previous_links = set()
    for i in range(len(data)):
        head, tail, time = data[i]
        previous_links.add((int(head), int(tail)))
    return previous_links 



def get_node2candidate(train_data, all_nodes, pri = False):
    head_node2candidate = dict()
    tail_node2candidate = dict()

    pri = True
    if pri:
        start_time = time.time()
        print('Start to build node2candidate')


    for i in range(len(train_data)):

        head, tail, not_in_use = train_data[i]
        head = int(head)
        tail = int(tail)
        if head not in head_node2candidate:
            head_node2candidate[head] = all_nodes

        if tail not in tail_node2candidate:
            tail_node2candidate[tail] = all_nodes



    if pri: 
        end_time = time.time()

        print('node2candidate built in' , str(end_time-start_time))
    return head_node2candidate, tail_node2candidate


def get_ranks(test_data,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate, pri=False, previous_links = None, bo = False):

    head_ranks = []
    tail_ranks = []
    head_lengths = []
    tail_lengths = []

    for interactioin in test_data:
        head_node, tail_node , time = interactioin
        head_node = int(head_node)
        tail_node = int(tail_node)
        if pri:
            print('--------------', head_node, tail_node, '---------------')


        if bo:
            if previous_links is not None: 
                if head_node in head_node2candidate and tail_node in tail_node2candidate and tail_node in head_node2candidate and head_node in tail_node2candidate and (head_node, tail_node) not in previous_links:
                    head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)
                    head_ranks.append(head_rank)
                    head_lengths.append(head_length)

                    tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)
                    tail_ranks.append(tail_rank)
                    tail_lengths.append(tail_length)
            else:

                if head_node in head_node2candidate and tail_node in tail_node2candidate and tail_node in head_node2candidate and head_node in tail_node2candidate:
                    head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)
                    head_ranks.append(head_rank)
                    head_lengths.append(head_length)

                    tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)
                    tail_ranks.append(tail_rank)
                    tail_lengths.append(tail_length)
        else:

            if previous_links is not None: 
                if head_node in head_node2candidate and tail_node in tail_node2candidate and (head_node, tail_node) not in previous_links:
                    head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)
                    head_ranks.append(head_rank)
                    head_lengths.append(head_length)

                    tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)
                    tail_ranks.append(tail_rank)
                    tail_lengths.append(tail_length)
            else:

                if head_node in head_node2candidate and tail_node in tail_node2candidate:
                    head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)
                    head_ranks.append(head_rank)
                    head_lengths.append(head_length)

                    tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)
                    tail_ranks.append(tail_rank)
                    tail_lengths.append(tail_length)

    return head_ranks, tail_ranks, head_lengths, tail_lengths












def train(args, data, num_nodes, model_save_dir):
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_negative = args.num_negative
    act = args.act
    transfer = args.transfer
    drop_p = args.drop_p
    if_propagation = args.if_propagation
    w = args.w
    is_att = args.is_att
    seed = args.seed
    reset_rep = args.reset_rep
    decay_method = args.decay_method
    nor = args.nor
    if_updated = args.if_updated
    weight_decay = args.weight_decay
    if_no_time = args.if_no_time
    threhold = args.threhold
    second_order = args.second_order
    num_iter = 4

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = data[0:int(len(data)*train_ratio)]
    validation_data = data[int(len(data)*train_ratio):int(len(data)*(train_ratio+valid_ratio))]
    test_data = data[int(len(data)*(train_ratio + valid_ratio)):len(data)]
    print('Data length: ', len(data))
    print('Train length: ', len(train_data))
    sampler = SequentialSampler(train_data)
    data_loader = DataLoader(train_data, batch_size, sampler = sampler)

    all_nodes = set(range(num_nodes))
    print('num_nodes',len(all_nodes))
    head_node2candidate, tail_node2candidate = get_node2candidate(train_data, all_nodes)



    model_save_dir = model_save_dir  + 'nt_' +str(if_no_time)+ '_wd_' + str(weight_decay) + '_up_' + str(if_updated) +'_w_' + str(w) +'_b_' + str(batch_size) + '_l_' + str(learning_rate) + '_tr_' + str(train_ratio) + '_nn_' +str(num_negative)+'_' + act + '_trans_' +str(transfer) + '_dr_p_' + str(drop_p) + '_prop_' + str(if_propagation) + '_att_' +str(is_att) + '_rp_' + str(reset_rep) + '_dcm_' + decay_method + '_nor_' + str(nor)
    if threhold is not None:
        model_save_dir = model_save_dir + '_th_' + str(threhold)

    if second_order:
        model_save_dir = model_save_dir + '_2hop'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    dyGnn = DyGNN(num_nodes,64,64,device, w,is_att ,transfer,nor,if_no_time, threhold,second_order, if_updated,drop_p, num_negative, act, if_propagation, decay_method )
    dyGnn.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,dyGnn.parameters()),lr = learning_rate, weight_decay=weight_decay)


    old_head_rank = num_nodes/2
    old_tail_rank = num_nodes/2

    for epoch in range(num_iter):
        print('epoch: ', epoch)
        print('Resetting time...')
        dyGnn.reset_time()
        print('Time reset')
        if reset_rep:

            dyGnn.reset_reps()
            print('reps reset')

        x = int(5000/batch_size)
        y = int(10000/batch_size)


        for i, interactions in enumerate(data_loader):

            # Compute and print loss.
            loss = dyGnn.loss(interactions)
            if i%x==0:
                #dyGnn.reset_reps()
                print(i,' train_loss: ', loss.item())

                if transfer:
                    head_reps = nn.Embedding.from_pretrained(dyGnn.transfer2head(dyGnn.node_representations.weight))
                    tail_reps = nn.Embedding.from_pretrained(dyGnn.transfer2tail(dyGnn.node_representations.weight))
                else:
                    head_reps = dyGnn.node_representations
                    tail_reps = dyGnn.node_representations

                head_reps = nn.Embedding.from_pretrained(nn.functional.normalize(head_reps.weight))
                tail_reps = nn.Embedding.from_pretrained(nn.functional.normalize(tail_reps.weight))


            if i%y==-1:

                if transfer:
                    head_reps = nn.Embedding.from_pretrained(dyGnn.transfer2head(dyGnn.node_representations.weight))
                    tail_reps = nn.Embedding.from_pretrained(dyGnn.transfer2tail(dyGnn.node_representations.weight))
                else:
                    head_reps = dyGnn.node_representations
                    tail_reps = dyGnn.node_representations

                head_reps = nn.Embedding.from_pretrained(nn.functional.normalize(head_reps.weight))
                tail_reps = nn.Embedding.from_pretrained(nn.functional.normalize(tail_reps.weight))

                head_ranks, tail_ranks, not_in_use, not_in_use2= get_ranks(validation_data,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate)
                head_ranks_numpy = np.asarray(head_ranks)
                tail_ranks_numpy = np.asarray(tail_ranks)
                print('head_rank mean: ', np.mean(head_ranks_numpy),' ; ', 'head_rank var: ', np.var(head_ranks_numpy))
                print('tail_rank mean: ', np.mean(tail_ranks_numpy),' ; ', 'tail_rank var: ', np.var(tail_ranks_numpy))


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()


        if transfer:
            head_reps = nn.Embedding.from_pretrained(dyGnn.transfer2head(dyGnn.node_representations.weight))
            tail_reps = nn.Embedding.from_pretrained(dyGnn.transfer2tail(dyGnn.node_representations.weight))
        else:
            head_reps = dyGnn.node_representations
            tail_reps = dyGnn.node_representations
        head_reps = nn.Embedding.from_pretrained(nn.functional.normalize(head_reps.weight))
        tail_reps = nn.Embedding.from_pretrained(nn.functional.normalize(tail_reps.weight))

        valid_loss = get_loss(validation_data, head_reps, tail_reps, device)
        head_ranks, tail_ranks, head_lengths, tail_lengths = get_ranks(validation_data, head_reps, tail_reps, device, head_node2candidate, tail_node2candidate)
        head_ranks_numpy = np.asarray(head_ranks)
        tail_ranks_numpy = np.asarray(tail_ranks)
        head_lengths_numpy = np.asarray(head_lengths)
        tail_lengths_numpy = np.asarray(tail_lengths)

        mean_head_rank = np.mean(head_ranks_numpy)
        mean_tail_rank = np.mean(tail_ranks_numpy)


        print('head_length mean: ', np.mean(head_lengths_numpy), ';', 'num_test: ', head_lengths_numpy.shape[0])
        print('tail_lengths mean: ', np.mean(tail_lengths_numpy), ';', 'num_test: ', tail_lengths_numpy.shape[0])
        print('head_rank mean: ', mean_head_rank,' ; ', 'head_rank var: ', np.var(head_ranks_numpy))
        print('tail_rank mean: ', mean_tail_rank,' ; ', 'tail_rank var: ', np.var(tail_ranks_numpy))
        print('reverse head_rank mean: ', np.mean(1/head_ranks_numpy))
        print('reverse tail_rank mean: ', np.mean(1/tail_ranks_numpy))
        print('head_rank HITS 100: ', (head_ranks_numpy<=100).sum())
        print('tail_rank_HITS 100: ', (tail_ranks_numpy<=100).sum())
        print('head_rank HITS 50: ', (head_ranks_numpy<=50).sum())
        print('tail_rank_HITS 50: ', (tail_ranks_numpy<=50).sum())
        print('head_rank HITS 20: ', (head_ranks_numpy<=20).sum())
        print('tail_rank_HITS 20: ', (tail_ranks_numpy<=20).sum())

        if mean_head_rank < old_head_rank or mean_tail_rank < old_tail_rank:
            model_save_path = model_save_dir + '/' + 'model_after_epoch_' + str(epoch) + '.pt'
            torch.save(dyGnn.state_dict(), model_save_path)
            print('model saved in: ', model_save_path)

            with open(model_save_dir + '/' + '0valid_results.txt','a') as f:
                f.write('epoch: ' + str(epoch) + '\n')
                f.write('head_rank mean: ' + str(mean_head_rank) + ' ; ' +  'head_rank var: ' + str(np.var(head_ranks_numpy)) + '\n')
                f.write('tail_rank mean: ' + str(mean_tail_rank) + ' ; ' +  'tail_rank var: ' + str(np.var(tail_ranks_numpy)) + '\n')
                f.write('head_rank HITS 100: ' + str ( (head_ranks_numpy<=100).sum()) + '\n')
                f.write('tail_rank_HITS 100: ' + str ( (tail_ranks_numpy<=100).sum()) + '\n')
                f.write('head_rank HITS 50: ' + str( (head_ranks_numpy<=50).sum()) + '\n')
                f.write('tail_rank_HITS 50: ' + str( (tail_ranks_numpy<=50).sum()) + '\n')
                f.write('head_rank HITS 20: ' + str( (head_ranks_numpy<=20).sum()) + '\n')
                f.write('tail_rank_HITS 20: ' + str( (tail_ranks_numpy<=20).sum()) + '\n')
                f.write('============================================================================\n')
            old_head_rank = mean_head_rank + 200
            old_tail_rank = mean_tail_rank + 200










if __name__ == '__main__':
    args = get_args()
    print(args)
    model_save_dir = 'saved_models/'
    if args.dataset == 'uci':
        data = Temporal_Dataset('Dataset/UCI_email_1899_59835/opsahl-ucsocial/out.opsahl-ucsocial',1,2)
        num_nodes = 1899
        model_save_dir = model_save_dir + 'UCI/'
        print('Train on UCI_message dataset')
        train(args, data, num_nodes, model_save_dir)   
    else:
        print('Please choose a dataset to run')

