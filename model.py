
import numpy as np
import pdb
from sklearn.decomposition import NMF
from numpy import power
import time
from data_handler import data_handler


class MATRI(object):
    def __init__(self, path, t, r, l):
        self.path = path
        self.t = t
        self.r = r
        self.l = l
        self.T, self.mu, self.x, self.y = data.load_data()


    def matri(self, iter):
        alpha = np.array([1,1,1])
        beta = np.zeros((1, 4 * t - 1))
        #for i in xrange(iter):


if __name__ == "__main__":
    data = data_handler("../data/advogato-graph-2000-02-25.dot",5)
    t = time.time()
    T, mu, x, y = data.load_data()
    print "Time for pre-processing is %f"%(time.time() - t)
    pdb.set_trace()
