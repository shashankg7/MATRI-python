from __future__ import print_function
import numpy as np
from numpy.linalg import norm
import pdb
from sklearn import linear_model
from sklearn.decomposition import NMF
from numpy import power
import time
from data_handler import data_handler
import sys


# TODO
# - Check alternative for Negative-Matrix-Factorization
# - Calculate RMSE
# - Parallelize

class MATRI(object):
    def __init__(self, t, r, l):
        print("Initializing MATRI...")
        self.max_iter = 1000
        self.t = t
        self.r = r
        self.l = l
        self.T, self.mu, self.x, self.y, self.k  = data.load_data()
        self.Z = np.zeros((len(self.k), 1, 4*self.t-1))
        self.Zt = np.zeros((len(self.k), 4*self.t-1, 1))
        print("Precomputing Zij....")
        for ind, (i,j) in enumerate(self.k):
            print("\rComputing " + str(ind+1) + " of " + str(len(self.k)) + " Zij matrices.",end="")
            sys.stdout.flush()
            self.Z[ind] = data.compute_prop(self.T, self.t, self.l, i, j)
            self.Zt[ind] = self.Z[ind].T
        print("\n")
        self.alpha = np.array([1,1,1])
        self.beta = np.zeros((1, 4 * t - 1))
        self._oldF = self.F = np.zeros((data.num_nodes, self.l))
        self._oldG = self.G = np.zeros((self.l, data.num_nodes))

    def updateCoeff(self, P):
        """ Update the Alpha and Beta weight vectors after each iteration """

        b = np.zeros((len(self.k)))
        for ind, (i, j) in enumerate(self.k):
            b[ind] = P[i, j]

        A = np.zeros((len(self.k), 4*self.t + 2))
        A[:,0] = self.mu
        for ind,(i,j) in enumerate(self.k):
            A[ind,1] = self.x[i]
            A[ind,2] = self.y[j]
            # Check dimension of Zt vs A[:,:,4:]
            # dim(Z[i,j]) = (1,4t-1)
            # dim(A[i,j]) = (1,4t+2)
            # and we skip the 1st 3 rows of A, therefore A's dimension available: (1,4t-1)
            # Check once. Seems good i guess?
            A[ind,3:] = self.Z[ind,:,:]
        clf = linear_model.Ridge(alpha = .5)
        clf.fit(A, b)
        self.alpha, self.beta = np.split(clf.coef_, [3])    # Split the matrix into concat of Alpha and Beta


    def converge(self, iterNO):
        """ Returns True if Converged, else return False """
        # Max iterations reached
        if iterNO >= self.max_iter:
            return True

        # Convergence is reached
        # EPS = np.finfo(float).eps
        EPS = 1
        E1 = np.absolute(norm(self.F) - norm(self._oldF))
        E2 = np.absolute(norm(self.G) - norm(self._oldG))
        if E1 < EPS and E2 < EPS:
            if iterNO != 1:   # Skip for the 1st iteration
                print("\rIteration: " + str(iterNO) + " FinalError ~ ("+str(E1)+","+str(E2)+")")
                return True

        self._oldF = self.F
        self._oldG = self.G
        print("\rIteration: "+str(iterNO)+" Error ~ ("+str(E1)+","+str(E2)+") (EPS:"+str(EPS)+")")
        return False


    def startMatri(self):
        """ Start the main MATRI algorithm """
        print(">> Starting MATRI",end='')
        P = np.zeros(self.T.shape)
        iter = 1
        while not self.converge(iter):
            print("Iteration:",iter,end='')
            for ind,(i,j) in enumerate(self.k):
                P[i, j] = self.T[i, j] - (np.dot(self.alpha, np.asarray([self.mu, self.x[i], self.y[j]]).T) + \
                        np.dot(self.beta, self.Zt[ind]))
            # ISSUE : sklearn's NMF accepts only non-neg matrices.
            # Currently taking absolute value of P, check other solution
            #self.F, self.G = data.mat_fact(np.absolute(P), self.r)
            self.F, self.G = data.mat_fact(P, self.r)
            for i,j in self.k:
                P[i, j] = self.T[i, j] - np.dot(self.F[i, :], self.G[j, :].T)

            # Update Alpha & Beta
            self.updateCoeff(P)
            iter += 1
        self.calcTrust()


    def calcTrust(self):
        """ Calculate final trust values for all; (u,v) belongs T """
        self.Tnew = np.zeros((self.num_nodes, self.num_nodes))
        for u in range(0, self.num_nodes):
            for v in range(0, self.num_nodes):
                self.Tnew[u,v] = np.dot(self.F[:,], self.G[v,:].T) + \
                                    np.dot(self.alpha.T, np.asarray([self.mu, self.x[u], self.y[v]])) \
                                        + np.dot(self.beta.T,self.Z[u,v])


    def RMSE(self):
        pass


if __name__ == "__main__":
    t = 6
    r = 10
    l = 10
    data = data_handler("data/advogato-graph-2000-02-25.dot", t)
    m = MATRI(t, r, l)
    m.startMatri()
