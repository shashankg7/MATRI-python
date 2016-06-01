
import networkx as nx
from scipy.io import savemat
import json
import numpy as np
import pdb
from sklearn.decomposition import NMF
from numpy import power
from datetime import datetime
import numpy
import pymf

class data_handler(object):
    def __init__(self, path, t):
        self.path = path
        self.t = t

    def mat_fact(self, X, l):
        """ X - matrix, l - latent factors
            Returns 2 factors of X, such that, dim(L) = n x r
                                               dim(R.T) = r x n
        """
        #model = NMF(n_components=l, init='random', random_state=0)
        nmf = pymf.NMF(X, num_bases=l)
        nmf.factorize()
        #L = model.fit_transform(X)
        L = nmf.W
        #R = model.components_
        R = nmf.H
        return L, R.T

    def mat_fact1(self, R, K):
        N = R.shape[0]
        P = np.random.rand(N, K)
        Q = np.random.rand(N, K)
        Q = Q.T
        steps = 5000
        alpha = 0.0002
        beta = 0.02
        for step in xrange(steps):
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                        for k in xrange(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = numpy.dot(P,Q)
            e = 0
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        for k in xrange(K):
                            e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
            if e < 0.001:
                break
        return P, Q

    def mat_pow(self, X, i):
        return power(X, i)

    def a2v(self, x):
        # Converts array of size (n,) to row vector - (1,n)
        return x.reshape(1, x.size)

    def compute_prop(self, T, t, l, i, j):
        L, R = self.mat_fact(T, l)
        # In paper R is N * L
        zij = []

        RtL = np.dot(R.T, L)
        for step in xrange(2, t+1):
            temp = self.mat_pow(RtL, step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), self.a2v(R[j,:]).T))

        LtR = np.dot(L.T, R)
        for step in xrange(1, t+1):
            temp = self.mat_pow(LtR, step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), self.a2v(L[j,:]).T))

        LtL = np.dot(L.T, L)
        RtR = np.dot(R.T, R)
        for step in xrange(1, t+1):
            temp = self.mat_pow(np.dot(LtL, RtR), step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), np.dot(LtL, self.a2v(R[j, :]).T)))
        for step in xrange(1, t+1):
            temp = self.mat_pow(np.dot(RtR, LtL), step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), np.dot(RtR, self.a2v(L[j, :]).T)))

        return np.asarray(zij).reshape(1, 4*t -1)

    def load_data(self):
        graph = nx.Graph(nx.drawing.nx_pydot.read_dot(self.path))
        nodes = graph.node.keys()
        edges = graph.edge
        self.num_nodes = len(nodes)
        self.num_edges = sum(map(lambda x:len(edges[x].keys()), edges))
        print "Nodes:",self.num_nodes, ", Edges:",self.num_edges
        node_to_index = dict(zip(nodes, range(len(nodes))))
        rating_map = {'"Observer"':0.1, '"Apprentice"':0.4, '"Journeyer"':0.7,
                      '"Master"':0.9}
        #rating_map = {'Observer':0.1, 'Apprentice':0.4, 'Journeyer':0.7,
        #              'Master':0.9}

        T = np.zeros((self.num_nodes, self.num_nodes))
        k = []
        for i, node in enumerate(nodes):
            edge_list = edges[node]
            for user in edge_list:
                T[i][node_to_index[user]] = rating_map[edge_list[user]['level']]
                k.append((i,node_to_index[user]))

        mu = np.sum(T)
        mu /= len(T[np.where(T > 0)])
        x = np.zeros(self.num_nodes)
        y = np.zeros(self.num_nodes)
        for i in xrange(0, self.num_nodes):
            x[i] = np.sum(T[i, :]) / len(T[i, np.where(T[i,:] > 0)])
            x[i] -= mu
            y[i] = np.sum(T[:, i]) / len(T[np.where(T[:, i] > 0), i])
            y[i] -= mu
        dp = np.linalg.matrix_power(T, self.t)
        return T, mu, x, y, k

if __name__ == "__main__":
    data = data_handler("data/advogato-graph-2011-06-23.dot",5)
    t = datetime.now()
    T, mu, x, y, k = data.load_data()
    t = (datetime.now() - t).total_seconds()
    print "Time for pre-processing is %fs"%(t)
    #pdb.set_trace()
