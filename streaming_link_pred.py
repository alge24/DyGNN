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
from train import rank, get_node2candidate, get_ranks, get_previous_links
from scipy.sparse import lil_matrix, find


def get_args():
    parser = argparse.ArgumentParser(description = 'Show description')
    parser.add_argument('-emb', '--emb_file', type = str,
    					help = 'emb_file', default = 'Dataset/UCI_email_1899_59835/opsahl-ucsocial/out.opsahl-ucsocial_line_200_64.emb')
    parser.add_argument('-emb2', '--emb_file2', type = str,
                        help = 'emb_file for graphsage', default = None)
    parser.add_argument('-emb3', '--emb_file3', type = str,
                        help = 'emb_file for DynGEM', default = None)
    parser.add_argument('-m', '--model_file', type = str,
                        help = 'DyGNN model file', required = True)
    parser.add_argument('-tr', '--train_ratio', type = float,
                        help = 'train_ratio', default = 0.8)
    parser.add_argument('-vr', '--valid_ratio', type = float,
                        help = 'valid_ratio', default = 0.01)
    parser.add_argument('-ter', '--test_ratio', type = float,
                        help = 'test_ratio', default = 0.1)
    parser.add_argument('-line', '--line', type = int,
                        help = 'line or not', default = 0)
    parser.add_argument('-p', '--pri', type = int,
                        help = 'print rank or not', default = 0)
    parser.add_argument('-trans', '--transfer', type = int,
                        help = 'transfer to head, tail representations', default = 0)
    parser.add_argument('-data', '--dataset', type = str,
                        help = 'which dataset to run', default = 'uci')
    parser.add_argument('-w', '--w', type = float,
                        help = 'w for decayer', default = 1)
    parser.add_argument('-ia', '--is_att', type = int,
                        help = 'use attention or not', default=0)
    parser.add_argument('-dc', '--decay_method', type = str,
                        help = 'decay_method', default = 'exp')
    parser.add_argument('-nor', '--nor', type = int ,
                        help = 'normalize or not', default = 0)
    parser.add_argument('-skip', '--skip', type = int,
                        help = 'skip existing links', default = 0)
    parser.add_argument('-val', '--val_file', type = str,
                        help = 'val_file for graphsage', default = None)
    parser.add_argument('-val3', '--val_file3', type = str,
                        help = 'val_file for dygem', default = None)
    parser.add_argument('-bo', '--bo', type = int,
                        help = 'bo', default = 1)
    parser.add_argument('-recon', '--test_reconstruct', type = int,
                        help = 'test_reconstruct', default = 0)
    parser.add_argument('-nt', '--if_no_time', type = int,
                        help = 'if no time interval information', default = 0)
    args = parser.parse_args()


    return args

def prepare_interaction_time(interactions, num_nodes, device):
    interaction_timestamp = lil_matrix((num_nodes,num_nodes),dtype = np.float32)
    recent_timestamp = torch.zeros((num_nodes, 1), dtype = torch.float, requires_grad = False).to(device)
    steps = len(interactions[:,0])
    print('Test: ',steps)
    for i in range(steps):
        head_index = int(interactions[i,0])
        tail_index = int(interactions[i,1])
        timestamp = interactions[i,2]
        current_t = torch.FloatTensor([timestamp]).view(-1,1).to(device)
        with torch.no_grad():
            recent_timestamp[[head_index, tail_index]] = current_t

        interaction_timestamp[head_index, tail_index] = current_t[0,0]
    return interaction_timestamp, recent_timestamp

def get_all_nodes(data):
    all_nodes = set()
    for i in range(len(data)):

        head, tail, not_in_use = data[i]
        head = int(head)
        tail = int(tail)
        all_nodes.add(head)
        all_nodes.add(tail)

    return all_nodes

def get_rank_new(interaction,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate, pri=False, previous_links = None, bo = False):
    head_rank = None
    tail_rank = None 
    head_length = None 
    tail_length =None
    head_node = interaction[0,0]
    tail_node = interaction[0,1]
    time = interaction[0,2]
    head_node = int(head_node)
    tail_node = int(tail_node)
    if pri:
        print('--------------', head_node, tail_node, '---------------')


    if bo:
        if previous_links is not None: 
            if head_node in head_node2candidate and tail_node in tail_node2candidate and tail_node in head_node2candidate and head_node in tail_node2candidate and (head_node, tail_node) not in previous_links:
                head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)


                tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)

        else:

            if head_node in head_node2candidate and tail_node in tail_node2candidate and tail_node in head_node2candidate and head_node in tail_node2candidate:
                head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)


                tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)

    else:

        if previous_links is not None: 
            if head_node in head_node2candidate and tail_node in tail_node2candidate and (head_node, tail_node) not in previous_links:
                head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)


                tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)

        else:

            if head_node in head_node2candidate and tail_node in tail_node2candidate:
                head_rank, head_length = rank(head_node, tail_node, head_node2candidate, head_reps, tail_reps, device,pri)


                tail_rank, tail_length = rank(tail_node, head_node, tail_node2candidate, tail_reps, head_reps, device)


    return head_rank, tail_rank, head_length, tail_length

def test(args,num_nodes, data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio
    line = args.line
    model_file = args.model_file
    pri = args.pri
    transfer = args.transfer
    w = args.w
    is_att = args.is_att
    decay_method = args.decay_method
    if_no_time = args.if_no_time
    nor = args.nor
    skip = args.skip
    train_data = data[0:int(len(data)*train_ratio)]
    train_end_time = train_data[-1][2]

    if skip:
        previous_links = get_previous_links(train_data)

    data_satrt_time = data[0][2]
    data_end_time = data[-1][2]
    print('Data Duration: ', str(data_end_time - data_satrt_time))


    validation_data = data[int(len(data)*train_ratio):int(len(data)*(train_ratio+valid_ratio))]
    print(test_ratio)
    if int(len(data)*(train_ratio+valid_ratio+test_ratio)) > len(data):
        test_data = data[int(len(data)*(train_ratio + valid_ratio)):len(data)]
    else:
        test_data = data[int(len(data)*(train_ratio+valid_ratio)):int(len(data)*(train_ratio+valid_ratio+test_ratio))]


    # valid_start_time = validation_data[0][2]
    # valid_end_time = validation_data[-1][2]
    test_start_time = test_data[0][2]
    test_end_time = test_data[-1][2]

    print('Train End Time: ',str( train_end_time), '; Test Start Time: ', str(test_start_time), 'Test End Time ', str(test_end_time),'; Gap: ', str(test_end_time- test_start_time))

    print('Data length: ', len(data))
    print('Train length: ', len(train_data))
    print('Valid length: ', len(validation_data))
    print('Test length: ', len(test_data))



    dyGnn = DyGNN(num_nodes,64,64, device,w,is_att,transfer, nor, if_no_time)
    dyGnn.eval()
    dyGnn.load_state_dict(torch.load(model_file))
    model_dict = dyGnn.state_dict()

    interaction_timestamp, recent_timestamp = prepare_interaction_time(train_data, num_nodes, device)
    dyGnn.interaction_timestamp = interaction_timestamp
    dyGnn.recent_timestamp = recent_timestamp

    head_ranks = []
    tail_ranks = []
    head_lengths = []
    tail_lengths = []
    print(test_data.shape)

    all_nodes = set(range(num_nodes))

    all_nodes = get_all_nodes(train_data)
    print(len(all_nodes))
    head_node2candidate, tail_node2candidate = get_node2candidate(train_data, all_nodes,pri)
    for interaction in test_data:
        interaction = np.reshape(interaction,(1,3))
        # print('test_len', interaction.shape)
        emb_emb = dyGnn.node_representations

        nor_ = nor
        if nor_==1:
            emb_emb = nn.Embedding.from_pretrained(nn.functional.normalize(emb_emb.weight))


        head_reps = emb_emb
        tail_reps = emb_emb

        if (line == 0 ) & (transfer == 1):
            head_reps = nn.Embedding.from_pretrained(dyGnn.transfer2head(emb_emb.weight))
            tail_reps = nn.Embedding.from_pretrained(dyGnn.transfer2tail(emb_emb.weight))


        if nor_ == 2:
            head_reps = nn.Embedding.from_pretrained(nn.functional.normalize(head_reps.weight))
            tail_reps = nn.Embedding.from_pretrained(nn.functional.normalize(tail_reps.weight))


        bo = args.bo
        if bo: 
            if skip:
                head_rank, tail_rank, head_length, tail_length = get_rank_new(interaction,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate, pri, previous_links,bo)
            else:
                head_rank, tail_rank, head_length, tail_length = get_rank_new(interaction,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate, pri, None, bo)
        else:

            if skip:
                head_rank, tail_rank, head_length, tail_length = get_rank_new(interaction,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate, pri, previous_links)
            else:
                head_rank, tail_rank, head_length, tail_length = get_rank_new(interaction,head_reps, tail_reps, device, head_node2candidate, tail_node2candidate, pri)

        if head_rank is not None:
            head_ranks.append(head_rank)
        if head_length is not None:
            head_lengths.append(head_length)

        if tail_rank is not None:
            tail_ranks.append(tail_rank)
        if tail_length is not None:
            tail_lengths.append(tail_length)

        dyGnn(interaction)



    head_ranks_numpy = np.asarray(head_ranks)
    tail_ranks_numpy = np.asarray(tail_ranks)
    head_lengths_numpy = np.asarray(head_lengths)
    tail_lengths_numpy = np.asarray(tail_lengths)
    print('head_length mean: ', np.mean(head_lengths_numpy), ';', 'num_test: ', head_lengths_numpy.shape[0])
    print('tail_lengths mean: ', np.mean(tail_lengths_numpy), ';', 'num_test: ', tail_lengths_numpy.shape[0])
    print('head_rank mean: ', np.mean(head_ranks_numpy),' ; ', 'head_rank var: ', np.var(head_ranks_numpy))
    print('tail_rank mean: ', np.mean(tail_ranks_numpy),' ; ', 'tail_rank var: ', np.var(tail_ranks_numpy))

    print('reverse head_rank mean: ', np.mean(1/head_ranks_numpy))
    print('reverse tail_rank mean: ', np.mean(1/tail_ranks_numpy))


    print('head_rank HITS 100: ', (head_ranks_numpy<=100).sum())
    print('tail_rank_HITS 100: ', (tail_ranks_numpy<=100).sum())
    print('head_rank HITS 50: ', (head_ranks_numpy<=50).sum())
    print('tail_rank_HITS 50: ', (tail_ranks_numpy<=50).sum())
    print('head_rank HITS 20: ', (head_ranks_numpy<=20).sum())
    print('tail_rank_HITS 20: ', (tail_ranks_numpy<=20).sum())

    print('MRR: ', (np.mean(1/head_ranks_numpy) +  np.mean(1/tail_ranks_numpy))/2 )
    print('Recall_20: ', ((head_ranks_numpy<=20).sum() +   (tail_ranks_numpy<=20).sum())/ head_lengths_numpy.shape[0]/2 )
    print('Recall_50: ', ((head_ranks_numpy<=50).sum() +   (tail_ranks_numpy<=50).sum())/  head_lengths_numpy.shape[0] /2 )











def main():
    args = get_args()
    print(args)
    if args.dataset == 'uci':
        data = Temporal_Dataset('Dataset/UCI_email_1899_59835/opsahl-ucsocial/out.opsahl-ucsocial',1,2)
        num_nodes = 1899
        print('Test on UCI dataset')
        test(args, num_nodes, data)
if __name__ == '__main__':
    main()
