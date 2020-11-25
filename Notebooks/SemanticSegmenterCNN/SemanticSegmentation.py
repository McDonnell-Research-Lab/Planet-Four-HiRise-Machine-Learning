import numpy as np
import copy
import tensorflow
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D, Lambda, concatenate,MaxPooling2D,UpSampling2D,Dropout,add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

def GetStats(GroundPosMask,PredPosMask):
    #calculate metrics
    TP = np.sum((GroundPosMask*PredPosMask).astype('float32'))
    TN = np.sum(((1-GroundPosMask)*(1-PredPosMask)).astype('float32'))
    FP = np.sum(((1-GroundPosMask)*PredPosMask).astype('float32'))
    FN = np.sum((GroundPosMask*(1-PredPosMask)).astype('float32'))
    TrueCoverage = (TP+FN+1e-12)/(TP+TN+FP+FN+1e-12)
    PredictedCoverage = (TP+FP+1e-12)/(TP+TN+FP+FN+1e-12)
    AreaAccuracy = (TP+FN+1e-12)/(TP+FP+1e-12)
    JI = (TP+1e-12)/(TP+FP+FN+1e-12)
    Dice = (2*TP+1e-12)/(2*TP+FP+FN+1e-12)
    Recall = (TP+1e-12)/(TP+FN+1e-12)
    Precision = (TP+1e-12)/(FP+TP+1e-12)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    AreaAccuracy = 1.0-np.absolute(FP-FN)/(TP+FN+1e-12)
    AreaAccuracyRevised = 1.0-np.absolute(FP-FN)/(TP+FN+FP+1e-12)
    AreaRatio = (TP+FN+1e-12)/(TP+FP+1e-12)
    return JI,Dice,AreaRatio,Recall,Precision,TP,TN,FP,FN

def metric_jaccard_coef_categorical_int(y_true, y_pred)
    smooth = 1e-12
    TargetClass = 1
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    predicted_mask = K.cast(K.equal(class_id_preds, TargetClass), 'float32')
    actual_mask = K.cast(K.equal(class_id_true, TargetClass), 'float32')
    #get the intersection and union
    total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1,2]) #sums over patches and space, this class
    total_class_sum = K.sum(actual_mask + predicted_mask, axis=[0,1,2]) #sums over patches and space, this class
    #calculate IOU for this class
    JI = (total_class_intersection + smooth) / (total_class_sum - total_class_intersection + smooth)
    return JI

def metric_precision(y_true, y_pred):
    TargetClass = 1
    smooth = 1e-12
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    predicted_mask = K.cast(K.equal(class_id_preds, TargetClass), 'float32')
    actual_mask = K.cast(K.equal(class_id_true, TargetClass), 'float32'
    #get the intersection 
    total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1,2]) #sums over patches and space, this clas
    #get the count of total predictions
    total_class_sum = K.sum(predicted_mask, axis=[0,1,2])
    Precision = (total_class_intersection + smooth) / (total_class_sum  + smooth)
    return Precision

def metric_recall(y_true, y_pred):
    TargetClass = 1
    smooth = 1e-12
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    predicted_mask = K.cast(K.equal(class_id_preds, TargetClass), 'float32')
    actual_mask = K.cast(K.equal(class_id_true, TargetClass), 'float32')
    #get the intersection 
    total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1,2]) #sums over patches and space, this class
    #get the count of total predictions
    total_class_sum = K.sum(actual_mask, axis=[0,1,2])
    Recall = (total_class_intersection + smooth) / (total_class_sum  + smooth)
    return Recall

def loss_jaccard_coef_categorical_one_minus(y_true, y_pred):
    smooth = 1e-12
    TargetClass = 1
    #Calculates a total loss equal to one minus the mean of the soft IOU over all classes.
    #inputs - assume 4 axes: [Patches,image rows, image columns, category indexes]
    #step 1: ignore the background class 
    y_true_no_background = y_true[:,:,:,TargetClass]
    y_pred_no_background = y_pred[:,:,:,TargetClass]
    #step 2: aggregate the union and intersection for each class, over all patches in a batch
    #2a. calculate the intersection for each class summed over all patches
    intersection = K.sum(y_true_no_background * y_pred_no_background, axis=[0,1,2])
    #2b. calculate the unions for each class summed over all patches
    sums = K.sum(y_true_no_background + y_pred_no_background, axis=[0,1,2]) 
    union = sums - intersection
    #step 3: calculate the mean IOU and subtract from 1 to get the loss function
    return 1.0- (intersection + smooth) / (union + smooth)

def UModule(inputs,num_filters,wd):
    x = BatchNormalization(center=True, scale=True)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=num_filters, 
               kernel_size=(3, 3), 
               padding='same', 
               kernel_initializer='he_uniform', 
               strides=1,
               use_bias=False,
               kernel_regularizer=l2(wd))(x)
    x = BatchNormalization(center=True, scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=num_filters, 
               kernel_size=(3, 3), 
               padding='same', 
               kernel_initializer='he_uniform', 
               strides=1,
               use_bias=False,
               kernel_regularizer=l2(wd))(x)
    return x

def model_unet_P4(num_channels_in_input=1,num_classes=2,base_filters=16,wd=0):
    inputs = Input(shape = (None,None,num_channels_in_input))
    #model input stage
    batch_norm0 = BatchNormalization()(inputs)
    conv0 = Conv2D(filters=base_filters, kernel_size=(7, 7), padding='same', use_bias=False, strides=1,
                   kernel_initializer='he_uniform',kernel_regularizer=l2(wd))(batch_norm0)
    
    #model main stage
    Module1 = UModule(inputs=conv0,num_filters=base_filters,wd=wd)
    Pool1 = MaxPooling2D(pool_size=(2, 2))(Module1)
    Module2 = UModule(inputs=Pool1,num_filters=base_filters*2,wd=wd)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(Module2)
    Module3 = UModule(inputs=Pool2,num_filters=base_filters*4,wd=wd)
    Pool3 = MaxPooling2D(pool_size=(2, 2))(Module3)
    Module4 = UModule(inputs=Pool3,num_filters=base_filters*8,wd=wd)
    Pool4 = MaxPooling2D(pool_size=(2, 2))(Module4)
    
    Module5 = UModule(inputs=Pool4,num_filters=base_filters*16,wd=wd)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(Module5), Module4])
    Module6 = UModule(inputs=up6,num_filters=base_filters*8,wd=wd)
    up7 = concatenate([UpSampling2D(size=(2, 2))(Module6), Module3])
    Module7 = UModule(inputs=up7,num_filters=base_filters*4,wd=wd)
    up8 = concatenate([UpSampling2D(size=(2, 2))(Module7), Module2])
    Module8 = UModule(inputs=up8,num_filters=base_filters*2,wd=wd)
    up9 = concatenate([UpSampling2D(size=(2, 2))(Module8), Module1])
    Module9 = UModule(inputs=up9,num_filters=base_filters,wd=wd)

    #model output stage
    ModuleOut = BatchNormalization(center=True, scale=True)(Module9)
    ModuleOut = Activation('relu')(ModuleOut)
    ModuleOut = Conv2D(filters=num_classes, kernel_size=(1, 1), padding='same', use_bias=False, strides=1,
                       kernel_initializer='he_uniform',kernel_regularizer=l2(wd))(ModuleOut)
    
    #consider an extra batch norm here?
    ModuleOut = Activation('softmax')(ModuleOut)
    
    model = Model(inputs=inputs, outputs=ModuleOut)

    return model


class P4_SegmenterDataGenerator(Sequence):
    def __init__(self, ImageLists,batch_size,patch_height,patch_width):
        self.ImageLists = ImageLists #this is a list of lists of training images at different scales
        self.batch_size = batch_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ordering = np.arange(len(self.ImageLists[0]))
        np.random.shuffle(self.ordering) 
    def on_epoch_end(self):
        np.random.shuffle(self.ordering)
    def __len__(self):
        return 35*int(np.floor(len(self.ImageLists[0])/self.batch_size))
    def __getitem__(self, index):
        #the data coming in has 4 channels (RGB plus mask). Do augmentation first and only separate for return
        imgs=[]
        for i in range(self.batch_size):
            #randomly choose from the scales available in the list of image lists
            WhichScale=np.random.randint(len(self.ImageLists))            
            img=self.ImageLists[WhichScale][self.ordering[(index*self.batch_size+i)%35]]
            #get a random location within the image
            start_height=np.random.randint(img.shape[0]-self.patch_height)
            start_width=np.random.randint(img.shape[1]-self.patch_width)
            #get the patch
            ThisPatch=copy.deepcopy(img[start_height:start_height+self.patch_height,start_width:start_width+self.patch_width,:])
            #add random flips and 90 degree rotations
            if np.random.rand(1)>0.5:
                ThisPatch=np.flipud(ThisPatch)
            if np.random.rand(1)>0.5:
                ThisPatch=np.fliplr(ThisPatch)
            if np.random.rand(1)>0.5:
                ThisPatch=np.rot90(ThisPatch)
            imgs.append(ThisPatch)
        #convert list to numpy
        imgs=np.stack(imgs,axis=0)
        #return the image channels and categorical mask representations separately
        return imgs[:,:,:,0:3],tensorflow.keras.utils.to_categorical(imgs[:,:,:,3],2)
      
#Tensorflow.keras hrnet model 
#Implementation based on: https://github.com/soyan1999/segmentation_hrnet_keras/blob/master/hrnet_keras.py
#Changes to the above implementation
#1. use he_normal initialisation
#2. change momentum from 0.1 to 0.9 - looks like they didn't realise Pytorch value of 0.1 = Tensorflow value of 0.9
#3. enable weight decay - I found this helped combat overfitting
#4. decoupled final softmax at end and inserted batch norm
#5. inserted initial batch norm

def conv(x, outsize, kernel_size, strides_=1, wd=1e-5, padding_='same', activation=None):
    return Conv2D(outsize, kernel_size, strides=strides_, padding=padding_, kernel_initializer='he_normal', use_bias=False, activation=activation,kernel_regularizer=l2(wd))(x)

def Bottleneck(x, size, downsampe=False,wd=1e-5):
    residual = x

    out = conv(x, size, 1, padding_='valid',wd=wd)
    out = BatchNormalization(epsilon=1e-5, momentum=0.9)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3,wd=wd)
    out = BatchNormalization(epsilon=1e-5, momentum=0.9)(out)
    out = Activation('relu')(out)

    out = conv(out, size * 4, 1,wd=wd)
    out = BatchNormalization(epsilon=1e-5, momentum=0.9)(out)

    if downsampe:
        residual = conv(x, size * 4, 1, padding_='valid',wd=wd)
        residual = BatchNormalization(epsilon=1e-5, momentum=0.9)(residual)

    out = add([out, residual])
    out = Activation('relu')(out)

    return out


def BasicBlock(x, size, downsampe=False,wd=1e-5):
    residual = x

    out = conv(x, size, 3,wd=wd)
    out = BatchNormalization(epsilon=1e-5, momentum=0.9)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3,wd=wd)
    out = BatchNormalization(epsilon=1e-5, momentum=0.9)(out)

    if downsampe:
        residual = conv(x, size, 1, padding_='valid',wd=wd)
        residual = BatchNormalization(epsilon=1e-5, momentum=0.9)(residual)

    out = add([out, residual])
    out = Activation('relu')(out)

    return out


def layer1(x,wd=1e-5):
    x = Bottleneck(x, 64, downsampe=True,wd=wd)
    x = Bottleneck(x, 64,wd=wd)
    x = Bottleneck(x, 64,wd=wd)
    x = Bottleneck(x, 64,wd=wd)

    return x


def transition_layer(x, in_channels, out_channels,wd=1e-5):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []

    for i in range(num_out):
        if i < num_in:
            if in_channels[i] != out_channels[i]:
                residual = conv(x[i], out_channels[i],3,wd=wd)
                residual = BatchNormalization(
                    epsilon=1e-5, momentum=0.9)(residual)
                residual = Activation('relu')(residual)
                out.append(residual)
            else:
                out.append(x[i])
        else:
            residual = conv(x[-1], out_channels[i], 3, strides_=2,wd=wd)
            residual = BatchNormalization(epsilon=1e-5, momentum=0.9)(residual)
            residual = Activation('relu')(residual)
            out.append(residual)

    return out


def branches(x, block_num, channels,wd=1e-5):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = BasicBlock(residual, channels[i],wd=wd)
        out.append(residual)
    return out


def fuse_layers(x, channels, multi_scale_output=True,wd=1e-5):
    out = []

    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        for j in range(len(channels)):
            if j > i:
                y = conv(x[j], channels[i], 1, padding_='valid',wd=wd)
                y = BatchNormalization(epsilon=1e-5, momentum=0.9)(y)
                y = UpSampling2D(size=2 ** (j - i))(y)
                residual = add([residual, y])
            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = conv(y, channels[i], 3, strides_=2,wd=wd)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.9)(y)
                    else:
                        y = conv(y, channels[j], 3, strides_=2,wd=wd)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.9)(y)
                        y = Activation('relu')(y)
                residual = add([residual, y])

        residual = Activation('relu')(residual)
        out.append(residual)

    return out


def HighResolutionModule(x, channels, multi_scale_output=True,wd=1e-5):
    residual = branches(x, 4, channels,wd=wd)
    out = fuse_layers(residual, channels,
                      multi_scale_output=multi_scale_output,wd=wd)
    return out


def stage(x, num_modules, channels, multi_scale_output=True,wd=1e-5):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = HighResolutionModule(out, channels, multi_scale_output=False,wd=wd)
        else:
            out = HighResolutionModule(out, channels,wd=wd)

    return out

def model_hrnet_P4(num_channels_in_input=1,num_classes=2,base_filters=16,wd=0):
    
    smaller=True
    
    #https://github.com/soyan1999/segmentation_hrnet_keras/blob/master/hrnet_keras.py
    channels_2 = [32, 64]
    channels_3 = [32, 64, 128]
    channels_4 = [32, 64, 128, 256]
    if smaller:
        channels_2 = [16, 32]
        channels_3 = [16, 32, 64]
        channels_4 = [16, 32, 64, 128]
    num_modules_2 = 1
    num_modules_3 = 4
    num_modules_4 = 3


    inputs = Input(shape = (None,None,num_channels_in_input))
    x = BatchNormalization(epsilon=1e-5, momentum=0.9)(inputs)
    x = conv(x, 64, 3, strides_=2,wd=wd)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
    x = conv(x, 64, 3, strides_=2,wd=wd)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)

    la1 = layer1(x,wd=wd)
    tr1 = transition_layer([la1], [256], channels_2,wd=wd)
    st2 = stage(tr1, num_modules_2, channels_2,wd=wd)
    tr2 = transition_layer(st2, channels_2, channels_3,wd=wd)
    st3 = stage(tr2, num_modules_3, channels_3,wd=wd)
    tr3 = transition_layer(st3, channels_3, channels_4,wd=wd)
    st4 = stage(tr3, num_modules_4, channels_4, multi_scale_output=False,wd=wd)
    up1 = UpSampling2D()(st4[0])
    up1 = conv(up1, 32, 3, wd=wd)
    up1 = BatchNormalization(epsilon=1e-5, momentum=0.9)(up1)
    up1 = Activation('relu')(up1)
    up2 = UpSampling2D()(up1)
    up2 = conv(up2, 32, 3, wd=wd)
    up2 = BatchNormalization(epsilon=1e-5, momentum=0.9)(up2)
    up2 = Activation('relu')(up2)
    
    final = conv(up2, num_classes, 1, padding_='valid',wd=wd)
    final = BatchNormalization(epsilon=1e-5, momentum=0.9)(final)
    final = Activation('softmax', name='Classification')(final)

    model = Model(inputs=inputs, outputs=final)

    return model
