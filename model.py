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
import copy
import sys,os
import pymf
import threading

# TODO
# - Parallelize

class MATRI(object):
    def __init__(self, t, r, l, max_itr):
        print("Initializing MATRI...")
        self.max_iter = max_itr
        self.t = t
        self.r = r
        self.l = l
        self.T, self.mu, self.x, self.y, self.k, self.deleted_edges, self.node_to_index, self.rating_map  = data.load_data()
        #pdb.set_trace()
        self.Z = np.zeros((len(self.k), 1, 4*self.t-1))
        self.Zt = np.zeros((len(self.k), 4*self.t-1, 1))

        file_name = "Z_train"
        if os.path.isfile(file_name + ".npy"):
            print("Loading Z from: " + file_name + ".npy")
            self.Z = np.load(file_name + ".npy")
        else:

            # THREADING MODULE
            # The Z array is not matching when calculated using threading, probably some issue with concurrency

            # THREADS=5
            # self.threads = [0 for i in range(THREADS)]
            # total = len(self.k)
            # each = total/THREADS
            # for t in range(0, THREADS):
            #     start = t*each
            #     end = (t+1)*each
            #     if t == THREADS - 1:
            #         end = total
            #     a = threading.Thread(name=str(t), target=self.calcZ, args=(start, end, total))
            #     a.setDaemon(True)
            #     a.start()

            # # JOIN ALL THREADS
            # main_thread = threading.currentThread()
            # for t in threading.enumerate():
            #     if t is main_thread:
            #         continue
            #     t.join()
            #     print('A thread ends')
            # # Check sum
            # s = 0
            # for i in self.threads:
            #     s += i
            # if s != total:
            #     print('Threading value not equal')
            #     exit()
            # else:
            #     print('Volla!! Threading says welcome')


            print("Precomputing Zij....")
            for ind, (i,j) in enumerate(self.k):
                print("\rComputing " + str(ind+1) + " of " + str(len(self.k)) + " Zij matrices.",end="")
                sys.stdout.flush()
                self.Z[ind] = data.compute_prop(self.T, self.t, self.l, i, j)
                self.Zt[ind] = self.Z[ind].T
            print("\n")
            np.save(file_name, self.Z)

        self.alpha = np.array([1,1,1])
        self.beta = np.zeros((1, 4 * self.t - 1))
        self._oldF = self.F = np.zeros((data.num_nodes, self.l))
        self._oldG = self.G = np.zeros((self.l, data.num_nodes))

    
    def alternatinUpdate(self, P, F, G, r):
        lamda = 0.1
        F1 = copy.deepcopy(F)
        for i in self.d.keys():
            a = self.d[i]
            d = np.zeros((len(a), 1))
            G1 = np.zeros((len(a), r))
            for j in xrange(len(a)):
                d[j] = self.T[i, a[j]]
                G1[j, :] = G[a[j], :]
            # Vectorize previous loop
            #d = self.T[i, a]
            #1[xrange(len(a)), :] = G[a,:]
            # TO-DO: Use sklearn's regression to find F1[i, :] instead
            temp = np.linalg.inv((np.dot(G1.T, G1) + lamda * np.eye(r))) 
            #pdb.set_trace()
            F1[i, :] = np.dot(np.dot(temp, G1.T), d).reshape(r,)
        return F1


    def mat_fact(self, X, r):
        print("Factorizing the matrices")
        F0 = np.zeros((self.T.shape[0], r))
        G0 = np.zeros((self.T.shape[0], r))
        F0[:] = 1/float(r)
        G0[:] = 1/float(r)
        EPS = 0.0001
        # pre-process self.k for alternatingUpate
        self.d = {}
        for i,j in self.k:
            if not self.d.has_key(i):
                self.d[i] = []
                self.d[i].append(j)
            else:
                self.d[i].append(j)
        iter = 1
        MAX_ITER = 200
        F = self.alternatinUpdate(X, F0, G0, r)
        G = self.alternatinUpdate(X.T, G0, F0, r)
        while norm(F - F0) > EPS and norm(G - G0) > EPS:
             if iter > MAX_ITER:
                return F, G
             F = self.alternatinUpdate(X, F0, G0, r)
             G = self.alternatinUpdate(X.T, G0, F0, r)
        return F, G
    

    def calcZ(self, start, end, total):
        """ Used to calculate Z only when we use threading """
        count = 0
        for ind, (i,j) in enumerate(self.k):
            if ind >= start and ind < end and ind < total:
                count += 1
                self.Z[ind] = data.compute_prop(self.T, self.t, self.l, i, j)
                self.Zt[ind] = self.Z[ind].T
        n = threading.currentThread().getName()
        self.threads[int(n)] = count


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
        clf = linear_model.Ridge(alpha = 0.1)
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
            if iterNO != 0:   # Skip for the 1st iteration
                print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
                return True

        self._oldF = copy.deepcopy(self.F)
        self._oldG = copy.deepcopy(self.G)
        print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
        return False


    def startMatri(self):
        """ Start the main MATRI algorithm """
        print(">> Starting MATRI")
        P = np.zeros(self.T.shape)
        
        # Compute Zij for test data
        self.Zij_test = self.compute_zij_test()

        # Join zij_test & zij_train
        self.Zij = self.join_zij()

        iter = 1
        while not self.converge(iter-1):
            print("Iteration: ",iter, end="")
            for ind,(i,j) in enumerate(self.k):
                P[i, j] = self.T[i, j] - (np.dot(self.alpha, np.asarray([self.mu, self.x[i], self.y[j]]).T) + \
                        np.dot(self.beta, self.Zt[ind]))
            # ISSUE : sklearn's NMF accepts only non-neg matrices.
            # Currently taking absolute value of P, check other solution
            #self.F, self.G = data.mat_fact(np.absolute(P), self.r)
            self.F, self.G = self.mat_fact(P, self.r)
            for i,j in self.k:
                P[i, j] = self.T[i, j] - np.dot(self.F[i, :], self.G[j, :].T)

            # Update Alpha & Beta
            self.updateCoeff(P)
            if iter % 5 == 0:
                self.calcTrust_test()
                print("RMSE after %d epoch is (on training) %f" %(iter, self.RMSE_test()))
            iter += 1


    def join_zij(self):
        """ Computes the full Zij by joining all the values"""
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        file2 = "Zij_all_save"
        if os.path.isfile(file2 + ".npy"):
            print("Loading FULL Zij from: " + file2 + ".npy")
            Zij = np.load(file2 + ".npy")
        else:
            # Copy test Zij
            ind = 1
            deleted_nodes = self.deleted_edges.keys()
            for i, node in enumerate(deleted_nodes):
                edge_list = self.deleted_edges[node]
                for user in edge_list:
                    u = self.node_to_index[node]
                    v = self.node_to_index[user]
                    Zij[u,v] = self.Zij_test[u,v]

            # Copy train Zij
            for ind, (i,j) in enumerate(self.k):
                Zij[i,j] = self.Z[ind]

            # Grab whatever you can !!
            np.save(file2, Zij)
        return Zij

    def compute_zij_test(self):
        """ Computes the full Zij on all n^2 values"""
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        file2 = "Zij_test"
        if os.path.isfile(file2 + ".npy"):
            print("Loading TEST Zij from: " + file2 + ".npy")
            Zij = np.load(file2 + ".npy")
        else:
            ind = 1
            deleted_nodes = self.deleted_edges.keys()
            for i, node in enumerate(deleted_nodes):
                edge_list = self.deleted_edges[node]
                for user in edge_list:
                    u = self.node_to_index[node]
                    v = self.node_to_index[user]
               
                    Zij[u,v] = data.compute_prop(self.T, self.t, self.l, u, v)
                    ind += 1
                    print("\rComputing (" + str(ind) + ")th Zij matrices. | TEST_DATASET",end="")
            print("\n")
            np.save(file2, Zij)
        return Zij


    def compute_zij(self):
        """ Computes the full Zij on all n^2 values"""
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        file2 = "Zij_all_save"
        if os.path.isfile(file2 + ".npy"):
            print("Loading FULL Zij from: " + file2 + ".npy")
            Zij = np.load(file2 + ".npy")
        else:
            ind = 1
            total = data.num_nodes*data.num_nodes
            for i in range(0, data.num_nodes):
                for j in range(0, data.num_nodes):
                    Zij[i,j] = data.compute_prop(self.T, self.t, self.l, i, j)
                    ind += 1
                    print("\rComputing " + str(ind) + " of " + str(total) + " Zij matrices.",end="")
            print("\n")
            np.save(file2, Zij)
        return Zij



    def calcTrust_test(self):
        """ Calculate final trust values for (u,v) belongs test-dataset """
        self.Tnew = np.zeros((data.num_nodes, data.num_nodes))

        deleted_nodes = self.deleted_edges.keys()
        for i, node in enumerate(deleted_nodes):
            edge_list = self.deleted_edges[node]
            for user in edge_list:
                u = self.node_to_index[node]
                v = self.node_to_index[user]
               
                # Used self.Z[u,v] instead of self.Z[u,v].T, due to numpy issues
                # Use G[v,:] instead of transpose
                A = np.dot(self.F[u,:], self.G[v,:])
                #pdb.set_trace()
                B = np.dot(self.alpha.T, np.asarray([self.mu, self.x[u], self.y[v]]))
                C = np.dot(self.beta, self.Zij[u,v].T)
                self.Tnew[u,v] = A + B + C



    def calcTrust(self):
        """ Calculate final trust values for all; (u,v) belongs T """
        self.Tnew = np.zeros((data.num_nodes, data.num_nodes))

        for u in range(0, data.num_nodes):
            for v in range(0, data.num_nodes):
                # Used self.Z[u,v] instead of self.Z[u,v].T, due to numpy issues
                # Use G[v,:] instead of transpose
                A = np.dot(self.F[u,:], self.G[v,:])
                #pdb.set_trace()
                B = np.dot(self.alpha.T, np.asarray([self.mu, self.x[u], self.y[v]]))
                C = np.dot(self.beta, self.Zij[u,v].T)
                self.Tnew[u,v] = A + B + C



    def RMSE(self):
        return np.sqrt(np.mean((self.Tnew-self.T)**2))



    def RMSE_test(self):
        """ Calculate RMSE only on the test data """
        tvalue_test = np.array([])
        tvalue_train = np.array([])
        deleted_nodes = self.deleted_edges.keys()
        for i, node in enumerate(deleted_nodes):
            edge_list = self.deleted_edges[node]
            for user in edge_list:
                n1 = self.node_to_index[node]
                n2 = self.node_to_index[user]
                tvalue_test = np.append(tvalue_test, self.rating_map[edge_list[user]['level']])
                tvalue_train = np.append(tvalue_train, self.Tnew[n1][n2])
        return np.sqrt(np.mean(np.square(np.subtract(tvalue_test, tvalue_train))))



if __name__ == "__main__":
    t = 6
    r = 10
    l = 10
    max_itr = 10000
    data = data_handler("data/advogato-graph-2000-02-25.dot", t)
    m = MATRI(t, r, l, max_itr)
    m.startMatri()
