import os
import sys
import numpy
from numpy import linalg as LA
from scipy import linalg
import h5py
import math


def makeList(input_file):
    return [item.rstrip() for item in open(input_file)]

def loadRandomWeight(file_name, dim):
    if os.path.isfile(file_name):
        sf = h5py.File(file_name)
        wgt = sf['wgt'][:]
        sf.close()
    else:
        sf = h5py.File(file_name, 'w')
        wgt = linalg.orth(2*numpy.random.rand(3+1, dim).transpose()-1).transpose()
        sf.create_dataset('wgt', data=wgt)
        sf.close()
    
    return wgt



# function to prepare sampling points
def prepSamplePoints(data_size, file_name, weight_name, dim, margin):
    # load existing one if it exists
    if not os.path.isfile(file_name):
       
        sample_points = numpy.zeros((data_size,3))
        for i in range(data_size):
            norm = 100
            randpoint = numpy.zeros((1,3))
            while norm > margin:
                randpoint = numpy.random.uniform(-margin,margin,3)
                norm = LA.norm(randpoint)
            sample_points[i] = randpoint
        b1 = loadRandomWeight(weight_name, dim, False)
        dval = numpy.std(sample_points)
        H1 = numpy.hstack((sample_points, dval* numpy.ones((sample_points.shape[0],1))))
        T1 = numpy.matmul(H1,b1)
        #relu
        T1 = numpy.maximum(T1, 0, T1)#np.tanh(T3*l3)
        mat1 = numpy.matmul(T1.transpose(),T1)
        val = dval*dval#(np.trace(mat1)/300)/12#*(0.001/300)
        mat1 = mat1 + numpy.eye(T1.shape[1]) * val
        #moore-penrose pseudoinverse
        pinvmat = numpy.matmul(LA.pinv(mat1),T1.transpose())
        #save
        sf = h5py.File(file_name, 'w')
        sf.create_dataset('points', data=sample_points)
        sf.create_dataset('mat', data=pinvmat)
        sf.close()