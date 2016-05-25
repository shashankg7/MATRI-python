
import networkx as nx
from scipy.io import savemat
import json
import numpy as np
import pdb
from sklearn.decomposition import NMF
from numpy import power
import time

class data_handler(object):
    def __init__(self, path, t):
        self.path = path
        self.t = t

    def mat_fact(self, X, l):
        # X - matrix, l - latent factors
        model = NMF(n_components=l, init='random', random_state=0)
        L = model.fit_transform(X)
        R = model.components_
        return L, R

    def mat_pow(self, X, i):
        return power(X, i)

    def a2v(self, x):
        # Converts array of size (n,) to row vector - (1,n)
        return x.reshape(1, x.size)

    def compute_prop(self, T, t, l, i, j):
        L, R = self.mat_fact(T, l)
        # In paper R is N * L
        R = R.T
        zij = []

        X = np.dot(R.T, L)
        for step in xrange(2, t+1):
            temp = self.mat_pow(X, step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), self.a2v(R[j,:]).T))

        X = np.dot(L.T, R)
        for step in xrange(1, t+1):
            temp = self.mat_pow(X, step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), self.a2v(L[j,:]).T))

        X1 = np.dot(L.T, L)
        for step in xrange(1, t+1):
            X = np.dot(X1, np.dot(R.T, R))
            temp = self.mat_pow(X, step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), np.dot(X1, self.a2v(R[j, :]).T)))

        X1 = np.dot(R.T, R)
        for step in xrange(1, t+1):
            X = np.dot(X1, np.dot(L.T, L))
            temp = self.mat_pow(X, step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), np.dot(X1, self.a2v(L[j, :]).T)))

        return np.asarray(zij).reshape(1, 4*t -1)

    def load_data(self):
        graph = nx.Graph(nx.drawing.nx_pydot.read_dot(self.path))
        nodes = graph.node.keys()
        edges = graph.edge
        num_nodes = len(nodes)
        num_edges = sum(map(lambda x:len(edges[x].keys()), edges))
        print "Nodes:",num_nodes, ", Edges:",num_edges
        node_to_index = dict(zip(nodes, range(len(nodes))))
        #rating_map = {'"Observer"':0.1, '"Apprentice"':0.4, '"Journeyer"':0.7,
        #              '"Master"':0.9}
        rating_map = {'Observer':0.1, 'Apprentice':0.4, 'Journeyer':0.7,
                      'Master':0.9}

        T = np.zeros((num_nodes, num_nodes))
        k = []
        for i, node in enumerate(nodes):
            edge_list = edges[node]
            for user in edge_list:
                T[i][node_to_index[user]] = rating_map[edge_list[user]['level']]
                k.append((i,node_to_index[user]))

        mu = np.sum(T)
        mu /= len(T[np.where(T > 0)])
        x = np.zeros(num_nodes)
        y = np.zeros(num_nodes)
        for i in xrange(0, num_nodes):
            x[i] = np.sum(T[i, :]) / len(T[i, np.where(T[i,:] > 0)])
            x[i] -= mu
            y[i] = np.sum(T[:, i]) / len(T[np.where(T[:, i] > 0), i])
            y[i] -= mu
        dp = np.linalg.matrix_power(T, self.t)
        t = self.compute_prop(T, 10, 50, 10, 10)
        #pdb.set_trace()
        return T, mu, x, y, k

if __name__ == "__main__":
    data = data_handler("data/advogato-graph-2000-02-25.dot",5)
    t = time.time()
    T, mu, x, y, k = data.load_data()
    print "Time for pre-processing is %f"%(time.time() - t)
    #pdb.set_trace()
