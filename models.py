from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Flatten,  GlobalMaxPooling1D, MaxPooling1D
from keras.models import  Model
import tensorflow as tf
from keras.layers import Lambda, concatenate


def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])



def defineModel(sample_size, dimsize, n_cls):

    input_pointsA = Input(shape=(sample_size, dimsize))
    x = Conv1D(512, 1, activation='relu')(input_pointsA)
    x = BatchNormalization()(x)
    x = Conv1D(512, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    
    global_feature = GlobalMaxPooling1D()(x)

    c = Dense(1024,activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    c = Dense(512,activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    c = Dense(256,activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    c = Dense(128,activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    prediction = Dense(n_cls, activation='softmax')(c)
    
    model = Model(inputs=input_pointsA, outputs=prediction)
    print(model.summary())
    return model

def defineModelPN(sample_size, dimsize, n_cls):

    input_pointsA = Input(shape=(sample_size,dimsize))
    input_pointsB = Input(shape=(sample_size,3))
    input_pointsC = Input(shape=(sample_size,3))
    x = concatenate([input_pointsA, input_pointsB, input_pointsC])
    x = Conv1D(512, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(512, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1,activation='relu')(x)
    x = BatchNormalization()(x)
    
    global_feature = GlobalMaxPooling1D()(x)

    c = Dense(1024,activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    c = Dense(512,activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    c = Dense(256,activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    c = Dense(128,activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.4)(c)
    prediction = Dense(n_cls, activation='softmax')(c)
    
    model = Model(inputs=[input_pointsA,input_pointsB,input_pointsC], outputs=prediction)
    print(model.summary())
    return model

def defineModelSegment(sample_size, dimsize, n_cls):
    input_points = Input(shape=(sample_size, dimsize))
    x = Conv1D(512, 1, activation='relu',
                    input_shape=(sample_size, dimsize))(input_points)
    x = BatchNormalization()(x)
    x = Conv1D(512, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)

    seg_part1 = x

    # global_feature
    global_feature = MaxPooling1D(2048)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': 2048})(global_feature)

    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    c = Conv1D(1024, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Conv1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Conv1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Conv1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Conv1D(n_cls, 1, activation='softmax')(c)

    model = Model(inputs=input_points, outputs=prediction)
    
    return model

def defineModelSegmentPN(sample_size, dimsize, n_cls):
    input_pointsA = Input(shape=(sample_size,dimsize))
    input_pointsB = Input(shape=(sample_size,3))
    input_pointsC = Input(shape=(sample_size,3))
    x = concatenate([input_pointsA, input_pointsB, input_pointsC])
    x = Conv1D(512, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(512, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)

    seg_part1 = x

    # global_feature
    global_feature = MaxPooling1D(2048)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': 2048})(global_feature)

    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    c = Conv1D(1024, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Conv1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Conv1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Conv1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Conv1D(n_cls, 1, activation='softmax')(c)

    model = Model(inputs=[input_pointsA,input_pointsB,input_pointsC], outputs=prediction)
    
    return model