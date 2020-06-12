
import argparse
import os
import h5py

import numpy
import cupy
from keras import optimizers
from keras.utils import to_categorical
from keras.models import load_model
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
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch')
    parser.add_argument('--dim', type=int, default=256, help='ELM weight size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4, help='Radius of sampling sphere')
    parser.add_argument('--train', type=bool, default=True, help='Conduct training')
    parser.add_argument('--normals', type=bool, default=True, help='Use normals for training')
    parser.add_argument('--sample_size', type=int, default=1024, help='Number of sampling points for distance field calculation')
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs


    DATA_DIR = #your directory
    SAVE_DIR = #your directory

    num_classes = 50
    train_inst = 12135
    test_inst = 2873

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
        #calculate distances
        print('---- test data ' + str(num) + ' ----')
        if not os.path.isfile(os.path.join(SAVE_DIR, svd_name)):
            print('**** Calculating Distance ****')
            process.calcDistField(os.path.join(SAVE_DIR, point_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR, svd_name))
        #convert them into ELM weights
        if not os.path.isfile(os.path.join(SAVE_DIR, weight_name)):
            print('**** Processing ELM ****')
            process.saveELM(os.path.join(SAVE_DIR,svd_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR, weight_name), os.path.join(SAVE_DIR, point_name), os.path.join(SAVE_DIR, basis_name), args.dim)

    if args.train:

        cat_file = os.path.join(DATA_DIR, 'synsetoffset2category.txt')
        cat_dict = {}
        with open(cat_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                cat_dict[ls[0]] = ls[1]
        cat_dict = {k:v for k,v in cat_dict.items()}

        seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        for cat in sorted(seg_classes.keys()):
            print(cat, seg_classes[cat])
        classes = dict(zip(cat_dict, range(len(cat_dict))))  
        print(classes)
        shape_ious = {cat:[] for cat in seg_classes.keys()}
        seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
    
    
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
                    y = HDF5Matrix(hdf5_file2, 'segment')

                    batchnm = int(numpy.ceil(size/batch_size))
                    blab = numpy.arange(batchnm)
                    numpy.random.shuffle(blab)
                    for itt in range(batchnm):
                        idx = blab[itt] * batch_size
                        if idx + batch_size >= size:
                            end = size
                        else:
                            end = idx + batch_size 
                        x1 = wgt[idx:end]
                        if args.normals:
                            x2 = pts[idx:end]
                            x3 = nms[idx:end]
                            xbatch = [x1, x2, x3]
                        else:
                            xbatch = x1
                        yield xbatch, to_categorical(y[idx:end],num_classes=num_classes)
    
        max_acc = 0
        max_iou = 0
        max_shapeious = None
        #prepare model
        overalldim = args.dim
        model_name = 'model' + str(overalldim) + 'segment.h5'
        if args.normals:
            mlp = models.defineModelSegmentPN(2048, args.dim, num_classes)
            adam = optimizers.Adam(lr=args.lr, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.000)
            mlp.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        else:
            mlp = models.defineModelSegment(2048, args.dim, num_classes)
            adam = optimizers.Adam(lr=args.lr, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.000)
            mlp.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # for multi gpu
        
        train_generator = generator(data_list_train, args.batch_size)
        test_generator = generator(data_list_test, args.batch_size)
        #iterate
       
        
        #training process
        for it in range(args.epochs):
            pred_val = numpy.zeros((test_inst,2048,num_classes))
            test_labels = numpy.zeros((test_inst,2048,num_classes))
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
                pred_val[curid:curid+batchend] = mlp.predict(x)
                test_labels[curid:curid+batchend] = y
                curid += batchend
            shape_ious = {cat:[] for cat in seg_classes.keys()}
            for i in range(test_inst):
                curpred = pred_val[i]
                segp = numpy.argmax(curpred,axis=1)
                segl = numpy.argmax(test_labels[i,:],axis=1).flatten()
                cat = seg_label_to_cat[segl[0]]
                
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (numpy.sum(segl==l) == 0) and (numpy.sum(segp==l) == 0): # part is not present, no prediction as well
                        part_ious[l-seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l-seg_classes[cat][0]] = numpy.sum((segl==l) & (segp==l)) / float(numpy.sum((segl==l) | (segp==l)))
                shape_ious[cat].append(numpy.mean(part_ious))
            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = numpy.mean(shape_ious[cat])
                
        
            mean_shape_ious = numpy.mean(numpy.fromiter(shape_ious.values(),dtype=float))
            for cat in sorted(shape_ious.keys()):
                
                print('eval mIoU of %s:\t %f' %(cat, shape_ious[cat]))
    
            print('eval mean mIoU: %f' % (mean_shape_ious))
            print('eval mean mIoU (all shapes): %f' % (numpy.mean(all_shape_ious)))

            if max_iou < numpy.mean(all_shape_ious):
                max_iou = numpy.mean(all_shape_ious)
                max_shapeious = shape_ious
            print('Maximum IoU: %f' % max_iou)
            if max_shapeious:
                for cat in sorted(max_shapeious.keys()):
                    print('eval mIoU of %s:\t %f' %(cat, max_shapeious[cat]))
                