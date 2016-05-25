
import numpy as np
import pdb
from sklearn.decomposition import NMF
from numpy import power
import time
from data_handler import data_handler


class MATRI(object):
    def __init__(self, t, r, l):
        self.t = t
        self.r = r
        self.l = l
        self.T, self.mu, self.x, self.y, self.k  = data.load_data()
        self.Z = np.zeros(self.T.shape)
        for i,j in self.k:
            self.Z[i, j] = data.compute_prop(self.T, self.t, self.l, i, j)


    def matri(self, iter):
        alpha = np.array([1,1,1])
        beta = np.zeros((1, 4 * t - 1))
        P = np.zeros(self.T.shape)

        #for i in xrange(iter):
        for i,j in self.k:
            P[i, j] = T[i, j] - (np.dot(alpha.T, np.asarray([self.mu, self.x[i], self.y[j]]).T) + np.dot(beta.T, self.Z[i,j]))

        F, G = data.mat_fact(P, self.r)
        for i,j in self.k:
            P[i, j] = T[i, j] - np.dot(F[i, :], G[j, :].T)

if __name__ == "__main__":
    t = 5
    r = 1000
    l = 1000
    data = data_handler("data/advogato-graph-2000-02-25.dot", t)
    #t = time.time()
    m = MATRI(t, r, l)
    #pdb.set_trace()
