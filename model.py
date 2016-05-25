
import numpy as np
from numpy.linalg import norm
import pdb
from sklearn import linear_model
from sklearn.decomposition import NMF
from numpy import power
import time
from data_handler import data_handler


class MATRI(object):
    def __init__(self, t, r, l):
        self.max_iter = 1000
        self.t = t
        self.r = r
        self.l = l
        self.T, self.mu, self.x, self.y, self.k  = data.load_data()
        self.Z = np.zeros(self.T.shape)
        for i,j in self.k:
            self.Z[i, j] = data.compute_prop(self.T, self.t, self.l, i, j)
        self.alpha = np.array([1,1,1])
        self.beta = np.zeros((1, 4 * t - 1))
        self._oldF = self.F = np.zeros((data.num_nodes, self.l))
        self._oldG = self.G = np.zeros((self.l, data.num_nodes))

    def updateCoeff(self, P):
        b = np.array(len(self.k))
        for ind, (i, j) in enumerate(self.k):
            b[ind] = P[i, j]

        A = np.zeros((len(self.k), 4*self.t + 2))
        A[:,1], A[:,2], A[:,3] = self.mu, self.x, self.y
        A[:,4:4*t+2] = self.Z.T
        clf = linear_model.Ridge(alpha = .5)
        clf.fit(A, b)
        self.alpha, self.beta = clf.coef_


    def converge(self, iterNO):
        """ Returns True if Converged, else return False """
        # Max iterations reached
        if iterNO >= self.max_iter:
            return True

        # Convergence is reached
        EPS = np.finfo(float).eps
        if np.absolute(norm(self.F) - norm(self._oldF)) < EPS and \
              np.absolute(norm(self.G) - norm(self._oldG)) < EPS:
            if iterNO == 1:   # Skip for the 1st iteration
                return False
            return True

        self._oldF = self.F
        self._oldG = self.G
        return False


    def startMatri(self):
        P = np.zeros(self.T.shape)
        iter = 1
        while not self.converge(iter):
            for i,j in self.k:
                P[i, j] = T[i, j] - (np.dot(self.alpha.T, np.asarray([self.mu, self.x[i], self.y[j]]).T) + np.dot(beta.T, self.Z[i,j]))

            self.F, self.G = data.mat_fact(P, self.r)
            for i,j in self.k:
                P[i, j] = T[i, j] - np.dot(F[i, :], G[j, :].T)

            self.alpha, self.beta = self.updateCoeff(P)
            iter += 1


if __name__ == "__main__":
    t = 5
    r = 1000
    l = 1000
    data = data_handler("data/advogato-graph-2000-02-25.dot", t)
    m = MATRI(t, r, l)
    m.startMatri()
