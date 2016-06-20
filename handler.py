from __future__ import division
import networkx as nx
from scipy.io import savemat
import json
import numpy as np
import pdb
from sklearn.decomposition import NMF
from numpy import power
from datetime import datetime
import numpy
from numpy import power
import pymf
import os
import copy
import sys

sys.path.append("./factorization")
from factorize_PYMF_SKLEARN import mat_fact_PYMF,  mat_fact_SKLEARN
from factorize_AF import mat_fact_AF
from factorize_GRAD import mat_fact_GRAD

class data_handler(object):
    def __init__(self, path, t):
        self.path = path
        self.t = t

    def a2v(self, x):
        # Converts array of size (n,) to row vector - (1,n)
        return x.reshape(1, x.size)

    def compute_prop(self, L, R, t, l, i, j):
        #L, R = mat_fact_pymf(T, l)
        #L, R, _d = mat_fact_AF(T, l, self.r, self.k)
        #L, R = mat_fact_grad(T, l)
        # Pre-compute L & R and pass it here


        # In paper R is N * L
        zij = []

        RtL = np.dot(R.T, L)
        for step in xrange(2, t+1):
            temp = power(RtL, step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), self.a2v(R[j,:]).T))

        LtR = np.dot(L.T, R)
        for step in xrange(1, t+1):
            temp = power(LtR, step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), self.a2v(L[j,:]).T))

        LtL = np.dot(L.T, L)
        RtR = np.dot(R.T, R)
        for step in xrange(1, t+1):
            temp = power(np.dot(LtL, RtR), step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), np.dot(LtL, self.a2v(R[j, :]).T)))
        for step in xrange(1, t+1):
            temp = power(np.dot(RtR, LtL), step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), np.dot(RtR, self.a2v(L[j, :]).T)))

        return np.asarray(zij).reshape(1, 4*t -1)

    def load_data(self):
        graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.path))
        nodes = graph.node.keys()
        edges = copy.deepcopy(graph.edge)
        #pdb.set_trace()
        REDUCE_DATA = True
        KEEP_EDGES = 500
        deleted_edges = {}
        if REDUCE_DATA:
            fileTrain = "data_train.txt"
            fileTest = "data_test.txt"
            if os.path.isfile(fileTrain) and os.path.isfile(fileTest):
                print("Loading the SPLIT dataset from file...")
                with open(fileTrain, 'r') as f:
                    edges = json.load(f)
                with open(fileTest, 'r') as f:
                    deleted_edges = json.load(f)
            else:
                print("Splitting dataset, Saving %d edges for testing" %(KEEP_EDGES))
                ind = 0
                while ind < KEEP_EDGES:
                    # Find a node
                    keys = edges.keys()
                    i = np.random.randint(0,len(keys))
                    n1 = keys[i]

                    # Find neighbour
                    neighbours = edges[n1]
                    keys2 = neighbours.keys()

                    # When the node has no neighbours then skip
                    if len(keys2) == 0:
                        continue
                    ind = ind + 1
                    j = np.random.randint(0,len(keys2))
                    n2 = keys2[j]

                    # Delete 'em
                    if not n1 in deleted_edges:
                        deleted_edges[n1] = {}
                    if not n2 in deleted_edges[n1]:
                        deleted_edges[n1][n2] = {}

                    deleted_edges[n1][n2].update(edges[n1].pop(n2))

                print("Saving dataset to files...")
                # Save only when we split the dataset
                with open('data_train.txt', 'w') as f:
                    json.dump(edges, f)
                with open('data_test.txt', 'w') as f:
                    json.dump(deleted_edges, f)

            deleted_num_edges = sum(map(lambda x:len(deleted_edges[x].keys()), deleted_edges))
            print('Deleted Edges = %d' %(deleted_num_edges))

        self.num_nodes = len(nodes)
        self.num_edges = sum(map(lambda x:len(edges[x].keys()), edges))
        print "Nodes:",self.num_nodes, ", Edges:",self.num_edges
        node_to_index = dict(zip(nodes, range(len(nodes))))
        #rating_map = {'"Observer"':0.1, '"Apprentice"':0.4, '"Journeyer"':0.7,
        #              '"Master"':0.9}
        rating_map = {'Observer':0.1, 'Apprentice':0.4, 'Journeyer':0.7,
                      'Master':0.9}

        T = np.zeros((self.num_nodes, self.num_nodes))
        k = []
        for i, node in enumerate(nodes):
            edge_list = edges[node]
            for user in edge_list:
                T[node_to_index[node]][node_to_index[user]] = rating_map[edge_list[user]['level']]
                k.append((node_to_index[node], node_to_index[user]))

        mu = np.sum(T)
        mu /= len(T[np.where(T > 0)])
        x = np.zeros(self.num_nodes)
        y = np.zeros(self.num_nodes)
        for ind, (i,j) in enumerate(k):
            x[i] = np.sum(T[i, :]) / len(np.where(T[i,:] > 0))
            x[i] -= mu
            y[j] = np.sum(T[:, j]) / len(np.where(T[:, j] > 0))
            y[j] -= mu
        #dp = np.linalg.matrix_power(T, self.t)
        return T, mu, x, y, k, deleted_edges, node_to_index, rating_map

if __name__ == "__main__":
    data = data_handler("dataset/advogato-graph-2011-06-23.dot",5)
    t = datetime.now()
    T, mu, x, y, k = data.load_data()
    t = (datetime.now() - t).total_seconds()
    print "Time for pre-processing is %fs"%(t)
