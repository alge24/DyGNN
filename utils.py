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
from nc import Node_classification






def get_loss(data, head_reps, tail_reps,device):

    head_list = list(data[:,0])
    tail_list = list(data[:,1])
    # head_reps = reps[head_list,:]
    # tail_reps = reps[tail_list,:]

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
    # print(len(candidates))
    candidates.append(true_candidate)
    # print(len(candidates))
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
    # head_node2nei = dict()
    # tail_node2nei = dict()
    pri = True
    if pri:
        start_time = time.time()
        print('Start to build node2candidate')

    # for node in all_nodes:
    #     head_node2candidate[node] = all_nodes
    #     tail_node2candidate[node] = all_nodes
    for i in range(len(train_data)):

        head, tail, not_in_use = train_data[i]
        head = int(head)
        tail = int(tail)
        if head not in head_node2candidate:
            head_node2candidate[head] = all_nodes

        if tail not in tail_node2candidate:
            tail_node2candidate[tail] = all_nodes



    # for node in head_node2nei:

    #     head_node2candidate[node] = all_nodes - head_node2nei[node]
    # for node in tail_node2nei:
    #     tail_node2candidate[node] = all_nodes - tail_node2nei[node]

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