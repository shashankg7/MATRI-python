from __future__ import print_function
import numpy as np
from numpy.linalg import norm
import pdb
from sklearn import linear_model
from sklearn.decomposition import NMF
from numpy import power
import time
from data_handler import data_handler
import sys, os

import sys,os
import pymf

# TODO
# - Check alternative for Negative-Matrix-Factorization
# - Calculate RMSE
# - Parallelize

class MATRI(object):
    def __init__(self, t, r, l, max_itr):
        print("Initializing MATRI...")
        self.max_iter = max_itr
        self.t = t
        self.r = r
        self.l = l
        self.T, self.mu, self.x, self.y, self.k  = data.load_data()
        self.Z = np.zeros((len(self.k), 1, 4*self.t-1))
        self.Zt = np.zeros((len(self.k), 4*self.t-1, 1))

        file_name = "Zij_save"
        if os.path.isfile(file_name + ".npy"):
            print("Loading Zij from: " + file_name + ".npy")
            self.Z = np.load(file_name + ".npy")
        else:
            print("Precomputing Zij....")
            for ind, (i,j) in enumerate(self.k):
                print("\rComputing " + str(ind+1) + " of " + str(len(self.k)) + " Zij matrices.",end="")
                sys.stdout.flush()
                self.Z[ind] = data.compute_prop(self.T, self.t, self.l, i, j)
                self.Zt[ind] = self.Z[ind].T
            np.save(file_name, self.Z)
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
        EPS = 0.000001
        E1 = np.absolute(norm(self.F) - norm(self._oldF))
        E2 = np.absolute(norm(self.G) - norm(self._oldG))
        if E1 < EPS and E2 < EPS:
            if iterNO != 1:   # Skip for the 1st iteration
                print("\rIteration: " + str(iterNO) + " FinalError ~ ("+str(E1)+","+str(E2)+")")
                return True

        self._oldF = self.F
        self._oldG = self.G
        #print("\rIteration: "+str(iterNO)+" Error ~ ("+str(E1)+","+str(E2)+") (EPS:"+str(EPS)+")")
        return False


    def startMatri(self):
        """ Start the main MATRI algorithm """
        print(">> Starting MATRI",end='')
        P = np.zeros(self.T.shape)
        iter = 1
        n_epochs = 10
        #self.calcTrust()
        #print("RMSE befor training is") 
        #print(self.RMSE())
        for epoch in xrange(n_epochs):
            iter = 1
            print("Epoch no %d" % epoch)
            while not self.converge(iter):
                #print("Iteration:",iter,end='')
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
            print("RMSE after trianing is")
            print(self.RMSE())


    def calcTrust(self):
        """ Calculate final trust values for all; (u,v) belongs T """
        self.Tnew = np.zeros((data.num_nodes, data.num_nodes))
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        file2 = "zij_all"
        if os.path.isfile(file2 + ".npy"):
            print("Loading FULL Zij from: " + file2 + ".npy")
            Zij = np.load(file2 + ".npy")
        else:
            for i in range(0, data.num_nodes):
                for j in range(0, data.num_nodes):
                    Zij[i,j] = data.compute_prop(self.T, self.t, self.l, i, j)
                    print("\rComputing " + str(i*10 + j + 1) + " of " + str(data.num_nodes*data.num_nodes) + " Zij matrices.",end="")
            print("\n")
            np.save(file2, Zij)


        for u in range(0, data.num_nodes):
            for v in range(0, data.num_nodes):
                #pdb.set_trace()
                # Used self.Z[u,v] instead of self.Z[u,v].T, due to numpy issues
                # Use G[v,:] instead of transpose
                A = np.dot(self.F[u,:], self.G[v,:])
                #pdb.set_trace()
                B = np.dot(self.alpha.T, np.asarray([self.mu, self.x[u], self.y[v]]))
                C = np.dot(self.beta, Zij[u,v].T)
                self.Tnew[u,v] = A + B + C

    def RMSE(self):
        return np.sqrt(np.mean((self.Tnew-self.T)**2))


if __name__ == "__main__":
    t = 6
    r = 10
    l = 10
    max_itr = 100
    data = data_handler("data/advogato-graph-2011-06-23.dot", t)
    m = MATRI(t, r, l, max_itr)
    m.startMatri()
