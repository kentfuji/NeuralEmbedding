
import argparse
import os
import h5py

import numpy
import cupy
from keras import optimizers
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
import threading

import utility
import process
import models

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g




if __name__ == '__main__':

    # mode argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch')
    parser.add_argument('--dim', type=int, default=256, help='ELM weight size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4, help='Radius of sampling sphere')
    parser.add_argument('--train', type=bool, default=True, help='Conduct training')
    parser.add_argument('--normals', type=bool, default=True, help='Use normals for training')
    parser.add_argument('--sample_size', type=int, default=1024, help='Number of sampling points for distance field calculation')
    parser.add_argument('--subset', type=int, default=1024, help='Subset for acceleration and robustness')
    args = parser.parse_args()


    DATA_DIR = '/home/www/current/data'#your directory
    SAVE_DIR = '/home/www/current/saved'#your directory

    
    num_classes = 40
    train_inst = 9840
    test_inst =2468
   
    #check if data preparation is necessary 
    data_list_train = utility.makeList(os.path.join(DATA_DIR, 'train_files.txt'))
    data_list_test = utility.makeList(os.path.join(DATA_DIR, 'test_files.txt'))
    
    
    #prepare samplings points
    basis_name = 'rands' + str(args.dim) + '.h5'
    point_name = 'points' + str(args.sample_size) + 'dim' + str(args.dim) + 'margin' + str(args.margin) + '.h5'
    utility.prepSamplePoints(args.sample_size, os.path.join(SAVE_DIR, point_name), os.path.join(SAVE_DIR, basis_name), args.dim, args.margin)
    for num in range(len(data_list_train)):
        original_name = data_list_train[num] + '.h5'
        svd_name = data_list_train[num] + 'margin' + str(args.margin) + 'svd.h5'
        weight_name = data_list_train[num] + 'margin' + str(args.margin) + 'dim' + str(args.dim) + '.h5'

        #calculate distances
        print('---- train data ' + str(num) + ' ----')
        if not os.path.isfile(os.path.join(SAVE_DIR, svd_name)):
            print('**** Calculating Distance ****')
            process.calcDistField(os.path.join(SAVE_DIR, point_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR, svd_name))
        #convert them into ELM weights
        if not os.path.isfile(os.path.join(SAVE_DIR, weight_name)):
            print('**** Processing ELM ****')
            process.saveELM(os.path.join(SAVE_DIR,svd_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR, weight_name), os.path.join(SAVE_DIR, point_name), os.path.join(SAVE_DIR, basis_name), args.dim)
   
    for num in range(len(data_list_test)):
        original_name = data_list_test[num] + '.h5'
        svd_name = data_list_test[num]+ 'margin' + str(args.margin) + 'svd.h5'
        weight_name = data_list_test[num]+ 'margin' + str(args.margin) + 'dim' + str(args.dim) + '.h5'
        print('---- test data ' + str(num) + ' ----')
        if not os.path.isfile(os.path.join(SAVE_DIR, svd_name)):
            print('**** Calculating Distance ****')
            process.calcDistField(os.path.join(SAVE_DIR, point_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR, svd_name))
        #convert them into ELM weights
        if not os.path.isfile(os.path.join(SAVE_DIR, weight_name)):
            print('**** Processing ELM ****')
            process.saveELM(os.path.join(SAVE_DIR,svd_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR, weight_name), os.path.join(SAVE_DIR, point_name), os.path.join(SAVE_DIR, basis_name), args.dim)

    if args.train:
        @threadsafe_generator
        def generator(data_list, batch_size):
            while True:
                numlab = numpy.arange(len(data_list))
                numpy.random.shuffle(numlab)
                for curnum in range(len(data_list)):
                    num = numlab[curnum]
                    hdf5_name = data_list[num]+ 'margin' + str(args.margin) + 'dim' + str(args.dim) + '.h5'
                    hdf5_file = os.path.join(SAVE_DIR, hdf5_name)
                    hdf5_name2 = data_list[num]+ '.h5'
                    hdf5_file2 = os.path.join(DATA_DIR, hdf5_name2)
                    wgt = HDF5Matrix(hdf5_file, 'data')
                    if args.normals:
                        pts = HDF5Matrix(hdf5_file2, 'data')
                        nms = HDF5Matrix(hdf5_file2, 'normal')
                    size = wgt.end
                    y = HDF5Matrix(hdf5_file2, 'label')

                    batchnm = int(numpy.ceil(size/batch_size))
                    blab = numpy.arange(batchnm)
                    numpy.random.shuffle(blab)
                    for itt in range(batchnm):
                        lab = numpy.arange(2048)
                        numpy.random.shuffle(lab)
                        lab = lab[:args.subset]
                        idx = blab[itt] * batch_size
                        if idx + batch_size >= size:
                            end = size
                        else:
                            end = idx + batch_size 
                        x1 = wgt[idx:end]
                        if args.normals:
                            x2 = pts[idx:end]
                            x3 = nms[idx:end]
                            xbatch = [x1[:,lab], x2[:,lab], x3[:,lab]]
                        else:
                            xbatch = x1[:,lab]
                        yield xbatch, to_categorical(y[idx:end],num_classes=num_classes)


        
        max_acc = 0.0
        #prepare model
        if args.normals:
            mlp = models.defineModelPN(args.subset, args.dim, num_classes)
            adam = optimizers.Adam(lr=args.lr, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.000)
            mlp.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        else:
            mlp = models.defineModel(args.subset, args.dim, num_classes)
            adam = optimizers.Adam(lr=args.lr, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.000)
            mlp.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        train_generator = generator(data_list_train, args.batch_size)
        test_generator = generator(data_list_test, args.batch_size)
        #training process
        curpred = numpy.zeros((test_inst,num_classes))
        test_labels_cat = numpy.zeros((test_inst,num_classes))
        for it in range(args.epochs):
            print('**** Epoch %03d ****' % (it))
            mlp.fit_generator(
                epochs=1,
                generator=train_generator,
                steps_per_epoch=int(numpy.ceil(train_inst / args.batch_size)),
                max_queue_size=10, 
                workers=1,  
                use_multiprocessing=False, 
                shuffle=False,verbose=1) 
            curid = 0
            for i in range(int(numpy.ceil(test_inst/args.batch_size))):
                x, y = next(test_generator)
                batchend = y.shape[0]
                curpred[curid:curid+batchend] = mlp.predict(x)
                test_labels_cat[curid:curid+batchend] = y
                curid += batchend

            pred_val = numpy.argmax(curpred, 1)
            test_labels = numpy.argmax(test_labels_cat, 1)
            correct = numpy.sum(pred_val.flatten() == test_labels.flatten())
            scores = float(correct / float(test_inst))
            print('Test accuracy: ', scores)
            if max_acc < scores:
                max_acc = scores
            print('Maximum accuracy: %f' % max_acc)