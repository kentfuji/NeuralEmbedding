import os
import sys
import numpy
import h5py
import cupy
import utility


def calcDistField(point_file, h5name, save_location):
    data_file = h5py.File(h5name)
    data = data_file['data'][:]
    data_dim = data.shape[0]
    data_file.close()
    ptfile = h5py.File(point_file)
    sample_points = ptfile['points'][:]
    ptfile.close()
    sample_size = sample_points.shape[0]

    #gpu parallelization
    memory_pool = cupy.get_default_memory_pool()
    pinned_memory_pool = cupy.get_default_pinned_memory_pool()
   
    distancesgpu = numpy.zeros((data_dim, data.shape[1], sample_size))
    x = cupy.asarray(sample_points)
    allpts = cupy.tile(x ,(data.shape[1], 1))
    blocks = int(numpy.ceil(sample_size*data.shape[1]/8192))
    del x
    print(blocks)
    yy = cupy.asarray(data)
    for inst in range(data_dim):
        if inst % 200 == 0:
            print(inst)
        y = yy[inst]
        
        xx = allpts + cupy.tile(y,(1,sample_size)).reshape(-1,3)
        xdot = cupy.sum(cupy.multiply(xx,xx),axis=1)
        dt = cupy.zeros((sample_size*data.shape[1],))
        for blk in range(blocks):
            idstart = int(blk * 8192)
            idend = int((blk + 1) * 8192)
           
            dists = cupy.tile(xdot[idstart:idend], (y.shape[0], 1)).transpose() - 2 * cupy.matmul(xx[idstart:idend], y.transpose()) + cupy.tile(cupy.sum(cupy.multiply(y,y),axis=1).transpose(), (xx[idstart:idend].shape[0],1))
            dt[idstart:idend] = cupy.amin(dists, axis=1)
            del dists
        dt = cupy.reshape(dt,(-1,sample_size))
        distancesgpu[inst] = cupy.asnumpy(dt)
        del dt
        del xx
        del xdot
    memory_pool.free_all_blocks()
    pinned_memory_pool.free_all_blocks()
    # save file
    saveh5 = h5py.File(save_location, 'w')
    saveh5.create_dataset('distances', data=distancesgpu)
    saveh5.close()


def saveELM(svd_file, original_file, final_file, point_file, weight_file, dim):
    file1 = h5py.File(svd_file)
    file2 = h5py.File(original_file)
    distances = file1['distances'][:]
    file1.close()
    file2.close()
    file3 = h5py.File(point_file)
    mat = file3['mat'][:]
    file3.close()
    surf_size = distances.shape[1]
    memory_pool = cupy.get_default_memory_pool()
    pinned_memory_pool = cupy.get_default_pinned_memory_pool()
    data_dim = distances.shape[0]
    tmp = numpy.zeros((data_dim, surf_size, dim))
    pinvmat = cupy.asarray(mat)
    for inst in range(data_dim):
        if inst % 200 == 0:
            print(inst)
        dt = cupy.asarray(distances[inst])
        res = cupy.matmul(pinvmat,dt.transpose())
        tmp[inst] = cupy.asnumpy(res.transpose())
        del dt
        del res
#        
    
    memory_pool.free_all_blocks()
    pinned_memory_pool.free_all_blocks()

    saveh5 = h5py.File(final_file, 'w')
    saveh5.create_dataset('data', data=tmp)
    saveh5.close()