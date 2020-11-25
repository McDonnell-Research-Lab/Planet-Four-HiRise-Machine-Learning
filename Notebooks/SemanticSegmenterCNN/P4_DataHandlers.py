import numpy as np
import pandas as pd
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None #use this to avoid a PIL warning when loading images
import os
import time
import glymur
import matplotlib.patches as patches
import cv2
from math import cos, degrees, pi, radians, sin
 
from p4tools import io #from https://github.com/michaelaye/p4tools

def AddHeightBorderVal(Im,Pixels):
    Padded = 128*np.ones((Im.shape[0]+2*Pixels,Im.shape[1],Im.shape[2]),'uint8')
    Padded[Pixels:Pixels+Im.shape[0],0:Im.shape[1],:] = Im
    return Padded  

def Load_all_HiRISE_images_crop_and_convert_to_8bits(data_folder):
    metadata_df = io.get_meta_data()
    Tiles_df = io.get_tile_coordinates()
    Image_list=[]
    OBSID_List=[]
    for index, row in metadata_df.iterrows():
        FileName = row['OBSERVATION_ID']+'_RGB.NOMAP.JP2'
        OBSID_List.append( row['OBSERVATION_ID'])
        #load the original JP2 images, which are 10 bits, stored as int16s
        jp2 = glymur.Jp2k(os.path.join(data_folder, FileName))
        original_img = jp2[:]
        original_img = 255.0*(original_img.astype('float32')/1024.0)
        original_img = original_img.astype('uint8')
        ThisImageTiles=Tiles_df[Tiles_df['obsid'].str.contains(row['OBSERVATION_ID'])]
        TiledSize_width = int(max(ThisImageTiles['x_hirise'])+min(ThisImageTiles['x_hirise']))
        TiledSize_height = int(max(ThisImageTiles['y_hirise'])+min(ThisImageTiles['y_hirise']))
        #remove the unlabelled borders
        #a small number of tiled sizes exceed the height of the original. Ignore any data outside the original size.
        original_img = original_img[0:min(TiledSize_height,original_img.shape[0]),0:min(TiledSize_width,original_img.shape[1]),:]
        Image_list.append(original_img)
    return Image_list,OBSID_List

#functions for concstruction markings from catalog data:
def rotate_vector(v, angle):
    rangle = radians(angle)
    rotmat = np.array([[cos(rangle), -sin(rangle)],[sin(rangle), cos(rangle)]])
    return rotmat.dot(v)

def get_arm_length(inside_half,distance):
    half = radians(inside_half)
    return distance / (cos(half) + sin(half))

def get_semicircle(semi_circle_center,radius,circle_base):
    # reverse order of arguments for arctan2 input requirements
    theta1 = degrees(np.arctan2(*circle_base[::-1]))
    theta2 = theta1 + 180
    wedge = patches.Wedge(semi_circle_center, radius, theta1, theta2)
    bb=wedge.get_path()
    return bb.vertices.astype('int64')

def fan_calcs(x,y,spread,angle,distance):
    base = (x,y)
    inside_half = spread / 2.0
    alpha = angle - inside_half
    beta = angle + inside_half 
    armlength = get_arm_length(inside_half,distance) # length of arms
    v1 = rotate_vector([armlength, 0], alpha)# first ar
    v2 = rotate_vector([armlength, 0], beta)# second arm
    cc = np.vstack((base + v1,base,base + v2)) # vector matrix, stows the 1D vectors row-wise
    circle_base = v1 - v2
    semi_circle_center = base + v2 + 0.5 * circle_base
    radius = 0.5 * np.linalg.norm(circle_base)
    return cc,circle_base,semi_circle_center,radius

def CreateFullImageMasks(ImageList):
    Fan_df = io.get_fan_catalog()
    metadata_df = io.get_meta_data()
    Blotch_df = io.get_blotch_catalog()
    Tiles_df = io.get_tile_coordinates()
    Images_names_in_order =[]
    for index, row in metadata_df.iterrows():
        Images_names_in_order.append(row['OBSERVATION_ID'])
    ImageAndMasksList=[]
    for i in range(len(ImageList)):
        #get the original JP2 image's width and height
        image_height,image_width, num_channels = ImageList[i].shape 
        #get the labelled width and height from tiling 
        ThisImageTiles = Tiles_df[Tiles_df['obsid'].str.contains(Images_names_in_order[i])]
        TiledSize_width = int(max(ThisImageTiles['x_hirise'])+min(ThisImageTiles['x_hirise']))
        TiledSize_height = int(max(ThisImageTiles['y_hirise'])+min(ThisImageTiles['y_hirise']))
        #note that a small number of tilings exceeded the height of the image
        #we will ignore any markings that fall outside the size of the original image
        #there shouldn't be any, because those areas would surely be blank
        fan_mask = np.zeros((image_height, image_width),'uint8')
        for index, row in Fan_df.iterrows():
            if row['obsid'][0:15] == Images_names_in_order[i]:
                pointy_points,circle_base,semi_circle_center,radius = fan_calcs(row['image_x'],
                                                                                row['image_y'],
                                                                                row['spread'],
                                                                                row['angle'],
                                                                                row['distance'])
                circle_points = get_semicircle(semi_circle_center,radius,circle_base)
                cv2.fillPoly(fan_mask, [pointy_points.astype('int32')], color=(1,1,1))
                cv2.fillPoly(fan_mask, [circle_points.astype('int32')], color=(1,1,1))
        blotch_mask = np.zeros((image_height, image_width),'uint8')
        for index, row in Blotch_df.iterrows():
            if row['obsid'][0:15] == Images_names_in_order[i]:
                blotch_mask=cv2.ellipse(blotch_mask,
                                        center=(int(row['image_x']), int(row['image_y'])),
                                        axes=(int(row['radius_1']),int(row['radius_2'])), 
                                        angle= row['angle'],
                                        startAngle=0,
                                        endAngle=360,
                                        color=(1,1,1),
                                        thickness=-1) 
        #combine
        markings_mask = (fan_mask+blotch_mask>=1).astype('uint8')
        #remove the unlabelled borders. A small number of tiled sizes exceed the orignal height. Ignore data outside this size:
        markings_mask = markings_mask[0:min(TiledSize_height,image_height),0:min(TiledSize_width,image_width)]
        ImageAndMasksList.append(np.concatenate((ImageList[i],np.expand_dims(markings_mask,axis=-1)),axis=-1))
    return ImageAndMasksList

def GetMarkingsCentres():
    metadata_df = io.get_meta_data()
    Fan_df = io.get_fan_catalog()
    Blotch_df = io.get_blotch_catalog()
    Images_names_in_order =[]
    for index, row in metadata_df.iterrows():
        Images_names_in_order.append(row['OBSERVATION_ID'])
    MarkingCount=0
    Markings_centres_df = pd.DataFrame()
    for i in range(len(Images_names_in_order)):
        ThisFan_df = Fan_df[Fan_df['obsid']==Images_names_in_order[i]]
        FanCount = ThisFan_df.shape[0]
        MarkingCount = MarkingCount + FanCount
        ThisBlotch_df = Blotch_df[Blotch_df['obsid']==Images_names_in_order[i]]
        BlotchCount = ThisBlotch_df.shape[0]
        MarkingCount = MarkingCount + BlotchCount
    #create a dataframe with the right number of rows. This init really speeds up the below
    Markings_centres_df = pd.DataFrame()
    Markings_centres_df['Image Name']=np.zeros(MarkingCount)
    Markings_centres_df['Image Name'] = Markings_centres_df['Image Name'].astype('str')
    MarkingCount = 0
    for i in range(len(Images_names_in_order)):
        ThisFan_df = Fan_df[Fan_df['obsid']==Images_names_in_order[i]]
        for index, row in ThisFan_df.iterrows():
            x = row['image_x']
            y = row['image_y']
            distance = row['distance']
            angle = row['angle']
            spread = row['spread']
            pointy_points,circle_base,semi_circle_center,radius = fan_calcs(x,y,spread,angle,distance)
            Markings_centres_df.at[MarkingCount,'Image Name'] = Images_names_in_order[i]
            Markings_centres_df.at[MarkingCount,'type'] = 'fan'
            Markings_centres_df.at[MarkingCount,'row'] = int(semi_circle_center[1])
            Markings_centres_df.at[MarkingCount,'column'] = int(semi_circle_center[0])
            MarkingCount = MarkingCount + 1
        ThisBlotch_df = Blotch_df[Blotch_df['obsid']==Images_names_in_order[i]]
        for index, row in ThisBlotch_df.iterrows():
            x = row['image_x']
            y = row['image_y']
            Markings_centres_df.at[MarkingCount,'Image Name'] = Images_names_in_order[i]
            Markings_centres_df.at[MarkingCount,'type'] = 'blotch'
            Markings_centres_df.at[MarkingCount,'row'] = y
            Markings_centres_df.at[MarkingCount,'column'] = x
            MarkingCount = MarkingCount + 1  
    Markings_centres_df['row'] = Markings_centres_df['row'].astype('int')
    Markings_centres_df['column'] = Markings_centres_df['column'].astype('int')
    return Markings_centres_df

def AddBorder(Im,Pixels):
    if len(Im.shape)==3:
        #have an image
        Padded = 128*np.ones((Im.shape[0]+2*Pixels,Im.shape[1]+2*Pixels,Im.shape[2]),'uint8')
        Padded[Pixels:Pixels+Im.shape[0],Pixels:Pixels+Im.shape[1],:] = Im
    elif len(Im.shape)==2:
        #have a mask
        Padded = np.zeros((Im.shape[0]+2*Pixels,Im.shape[1]+2*Pixels),'uint8')
        Padded[Pixels:Pixels+Im.shape[0],Pixels:Pixels+Im.shape[1]] = Im
    return Padded  
                               
def CreateListOfThreeScales(ImageList,DoMasks=True):
    #Scale augmentation: Create three copies of image, resized to scale 1, 0.5 an 0.25
    #use bicubic resizing for images but default nearest neighbour for masks
    metadata_df = io.get_meta_data()
    Scales =  metadata_df['map_scale'].values                          
    CC=0
    BorderSize_1 = 16
    BorderSize_05 = 32
    BorderSize_025 = 64
    ImageList_scale_1=[]
    ImageList_scale_05=[]
    ImageList_scale_025=[]
    for Im in ImageList:
        if Scales[CC]==1.0:
            img = AddBorder(Im[:,:,0:3],BorderSize_1)
            if DoMasks:
                mask = np.expand_dims(AddBorder(Im[:,:,3],BorderSize_1),axis=-1)
                ImageList_scale_1.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_1.append(img)
            img =np.array(Image.fromarray(Im[:,:,0:3],mode='RGB').resize([int(Im.shape[1]*2),int(Im.shape[0]*2)],resample=PIL.Image.BICUBIC))
            img = AddBorder(img,BorderSize_05)
            if DoMasks:
                mask =np.array(Image.fromarray(Im[:,:,3],mode='L').resize([int(Im.shape[1]*2),int(Im.shape[0]*2)]))
                mask = np.expand_dims(AddBorder(mask,BorderSize_05),-1)
                ImageList_scale_05.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_05.append(img)
            img =np.array(Image.fromarray(Im[:,:,0:3],mode='RGB').resize([int(Im.shape[1]*4),int(Im.shape[0]*4)],resample=PIL.Image.BICUBIC))
            img = AddBorder(img,BorderSize_025)
            if DoMasks:
                mask =np.array(Image.fromarray(Im[:,:,3],mode='L').resize([int(Im.shape[1]*4),int(Im.shape[0]*4)]))
                mask = np.expand_dims(AddBorder(mask,BorderSize_025),-1)
                ImageList_scale_025.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_025.append(img)
        elif Scales[CC]==0.5:
            img = AddBorder(Im[:,:,0:3],BorderSize_05)
            if DoMasks:
                mask = np.expand_dims(AddBorder(Im[:,:,3],BorderSize_05),axis=-1)
                ImageList_scale_05.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_05.append(img)
            img =np.array(Image.fromarray(Im[:,:,0:3],mode='RGB').resize([int(Im.shape[1]*2),int(Im.shape[0]*2)],resample=PIL.Image.BICUBIC))
            img = AddBorder(img,BorderSize_025)
            if DoMasks:
                mask =np.array(Image.fromarray(Im[:,:,3],mode='L').resize([int(Im.shape[1]*2),int(Im.shape[0]*2)]))
                mask = np.expand_dims(AddBorder(mask,BorderSize_025),-1)
                ImageList_scale_025.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_025.append(img)
            img =np.array(Image.fromarray(Im[:,:,0:3],mode='RGB').resize([int(Im.shape[1]/2),int(Im.shape[0]/2)],resample=PIL.Image.BICUBIC))
            img = AddBorder(img,BorderSize_1)
            if DoMasks:
                mask =np.array(Image.fromarray(Im[:,:,3],mode='L').resize([int(Im.shape[1]/2),int(Im.shape[0]/2)]))
                mask= np.expand_dims(AddBorder(mask,BorderSize_1),-1)
                ImageList_scale_1.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_1.append(img)
        elif Scales[CC]==0.25:
            img = AddBorder(Im[:,:,0:3],BorderSize_025)
            if DoMasks:
                mask = np.expand_dims(AddBorder(Im[:,:,3],BorderSize_025),axis=-1)
                ImageList_scale_025.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_025.append(img)
            img =np.array(Image.fromarray(Im[:,:,0:3],mode='RGB').resize([int(Im.shape[1]/2),int(Im.shape[0]/2)],resample=PIL.Image.BICUBIC))
            img = AddBorder(img,BorderSize_05)
            if DoMasks:
                mask = np.array(Image.fromarray(Im[:,:,3],mode='L').resize([int(Im.shape[1]/2),int(Im.shape[0]/2)]))
                mask = np.expand_dims(AddBorder(mask,BorderSize_05),-1)
                ImageList_scale_05.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_05.append(img)
            img =np.array(Image.fromarray(Im[:,:,0:3],mode='RGB').resize([int(Im.shape[1]/4),int(Im.shape[0]/4)],resample=PIL.Image.BICUBIC))
            img = AddBorder(img,BorderSize_1)
            if DoMasks:         
                mask = np.array(Image.fromarray(Im[:,:,3],mode='L').resize([int(Im.shape[1]/4),int(Im.shape[0]/4)]))
                mask = np.expand_dims(AddBorder(mask,BorderSize_1),-1)
                ImageList_scale_1.append(np.concatenate((img,mask),axis=-1))
            else:
                ImageList_scale_1.append(img)
        CC=CC+1
    ImageLists=[ImageList_scale_1,ImageList_scale_05,ImageList_scale_025]
    return ImageLists

def Load_P4_Data_PreSaved(data_folder,region_names,metadata_df,Tiles_df):
    #load images and masks. Remove regions not labelled.
    ImageList = []
    Scales=[]
    ImageNames=[]
    RegionNames=[]
    filenames=[]
    for filename in os.listdir(data_folder):
        filename_without_ext = os.path.splitext(filename)[0]
        if filename_without_ext[0:3] != 'ESP':
            continue
        region = metadata_df.at[filename_without_ext,'roi_name']
        if region in region_names:
            ImageNames.append(filename_without_ext)
            filenames.append(filename)
    ImageNames.sort()
    filenames.sort()
    Count = 0
    for filename_without_ext in ImageNames:
        if filename_without_ext[0:3] != 'ESP':
            continue
        region = metadata_df.at[filename_without_ext,'roi_name']
        if region in region_names:
            filename = filenames[Count]
            Scale = metadata_df.at[filename_without_ext,'map_scale']
            ThisImageTiles=Tiles_df[Tiles_df['obsid'].str.contains(filename_without_ext[0:15])]
            TiledSize_width = int(max(ThisImageTiles['x_hirise'])+min(ThisImageTiles['x_hirise']))
            TiledSize_height = int(max(ThisImageTiles['y_hirise'])+min(ThisImageTiles['y_hirise']))
            #load the original image and mask
            ThisIm=np.load(os.path.join(data_folder, filename))
            ImageList.append(ThisIm)
            Scales.append(Scale)
            RegionNames.append(region)
            Count = Count+ 1
    return ImageList,Scales,ImageNames,RegionNames
