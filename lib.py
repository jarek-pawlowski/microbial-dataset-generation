import numpy as np
import random as rnd
import time
from enum import Enum

from skimage import color
from skimage.filters import unsharp_mask
from skimage.morphology import dilation, square
from skimage.restoration import  estimate_sigma, denoise_nl_means
from skimage.segmentation import checkerboard_level_set, chan_vese

import cv2
from scipy.spatial import KDTree

import matplotlib.cm as cm

class Parameters(Enum):
    box = 1
    scanner = 2
    box_aeruginosa = 3
    
category_id_dict = {'S.aureus': "1", 'B.subtilis': "2", 'P.aeruginosa': "3", 'E.coli': "4", 'C.albicans': "6", 'Defect': "0", 'Contamination': "0"}

parameters_family = Parameters.box  # box or scanner or aeruginosa
whole_dish = False  # True for a whole dish generation, False if smaller patches should be generated

if parameters_family.value == 1:

    translate_factor = 4000./700
    scalling_factor = .3
    adjacency_threshold = .01
    dark_regions_threshold = 25
    dark_matter_threshold = 5
    colonies_threshold = 30
    image_size = 1024
    dish_radius = 512 # in pixels
    remove_labels_flag = True
    patches_dirs = {"aureus":"1",\
                    "subtilis":"2",\
                    "aeruginosa":"3",\
                    "coli":"4", \
                    "albicans":"6"}
    style_dirs = {"aureus":"1",\
                  "subtilis":"2",\
                  "aeruginosa":"3",\
                  "coli":"4",\
                  "albicans":"6"}

if parameters_family.value == 2:

    translate_factor = 2048./700
    scalling_factor = .54
    adjacency_threshold = .01
    dark_regions_threshold = 25
    dark_matter_threshold = 5
    colonies_threshold = 30
    image_size = 1024
    dish_radius = 475 # in pixels
    remove_labels_flag = False
    patches_dirs = {"aureus":"1",\
                    "aeruginosa":"3",\
                    "coli":"4"}
    style_dirs = {"aureus":"1",\
                  "aeruginosa":"3",\
                  "coli":"4"}

if parameters_family.value == 3:

    translate_factor = 4000./700
    scalling_factor = .3
    adjacency_threshold = .01
    dark_regions_threshold = 25
    dark_matter_threshold = 5
    colonies_threshold = 30
    image_size = 1024
    dish_radius = 512 # in pixels
    remove_labels_flag = True
    patches_dirs = {"aeruginosa":"3"}
    style_dirs = {"aeruginosa":"3b"}


if whole_dish == False: 
    scalling_factor = 1.

class Coordinate:
    def __init__(self, coordinate):
        self.x1 = coordinate[0]
        self.x2 = coordinate[1]
        self.y1 = coordinate[2]
        self.y2 = coordinate[3]
    def __mul__(self, factor):
        return(Coordinate([self.x1*factor,self.x2*factor,self.y1*factor,self.y2*factor]))

    __rmul__ = __mul__

    def to_int(self):
        self.x1 = int(self.x1)
        self.x2 = int(self.x2)
        self.y1 = int(self.y1)
        self.y2 = int(self.y2)


def adjacency_matrix_between_patches(coordinates):
    """
    matrix describing mutual intersection between (rectangular) patches
    """
    adj = np.zeros((len(coordinates),len(coordinates)),dtype=np.float16)
    for ic, c0 in enumerate(coordinates):
        for jc, c1 in enumerate(coordinates[ic+1:]):
            xx = 0.
            if c1.x1 < c0.x1:
                if c1.x2  > c0.x1:
                    if c1.x2 < c0.x2 : 
                        xx = c1.x2 - c0.x1
                    else: xx = c0.x2 - c0.x1         
            else:
                if c1.x1 < c0.x2 :
                    if c1.x2  < c0.x2 : 
                        xx = c1.x2 - c1.x1
                    else: xx = c0.x2 - c1.x1
            yy = 0.
            if c1.y1 < c0.y1:
                if c1.y2  > c0.y1:
                    if c1.y2 < c0.y2 : 
                        yy = c1.y2 - c0.y1
                    else: yy = c0.y2 - c0.y1         
            else:
                if c1.y1 < c0.y2 :
                    if c1.y2  < c0.y2 : 
                        yy = c1.y2  - c1.y1
                    else: yy = c0.y2  - c1.y1

            if np.sqrt(xx*yy/(c0.x2-c0.x1)/(c0.y2-c0.y1)) > adjacency_threshold \
                    or np.sqrt(xx*yy/(c1.x2-c1.x1)/(c1.y2-c1.y1)) > adjacency_threshold:
                adj[ic,jc+ic+1] = 1.
    return adj

def del_multiple(list_to_del, indices):
    return [val for i, val in enumerate(list_to_del) if i not in indices]

def rand_position(occupation_matrix, patch_size):
    patch_radius = np.sqrt(patch_size[0]**2+patch_size[1]**2)/2
    if patch_radius > dish_radius*.7: 
        raise Exception('too large patch to be placed')
    trial_occupation = np.zeros_like(occupation_matrix)
    watchdog = 0
    while True:
        while True:
            x = rnd.randint(-dish_radius,dish_radius)
            y = rnd.randint(-dish_radius,dish_radius)
            if whole_dish:
                if x**2+y**2 < dish_radius**2: break
            else: 
                break
        if whole_dish:
            if np.sqrt(x**2+y**2) + patch_radius > dish_radius: continue
        else:
            if np.abs(x) + patch_radius > dish_radius: continue
            if np.abs(y) + patch_radius > dish_radius: continue
        x += int(image_size/2)
        y += int(image_size/2)
        x1 = x-int(patch_size[0]/2)
        x2 = x1+patch_size[0]
        y1 = y-int(patch_size[1]/2)
        y2 = y1+patch_size[1]
        trial_occupation[:,:] = 0
        trial_occupation[x1:x2,y1:y2] = 1 
        if not np.any(np.multiply(occupation_matrix,trial_occupation)): break
        watchdog += 1
        if watchdog > 100: raise Exception('too many bacterias on dish')
    return [x, y]

def blend_patch(occupation_matrix, dish, patch, position):
    x1 = position[0]-int(patch.shape[0]/2)
    x2 = x1+patch.shape[0]
    y1 = position[1]-int(patch.shape[1]/2)
    y2 = y1+patch.shape[1]
    alpha = patch[:,:,3]/255
    for i in [0,1,2]:
        dish[x1:x2,y1:y2,i] = dish[x1:x2,y1:y2,i]*(1.-alpha[:,:]) + patch[:,:,i]*alpha[:,:]
    occupation_matrix[x1:x2,y1:y2] = 1

def segmentation_mask(segmentation_matrix, patch, position):
    x1 = position[0]-int(patch.shape[0]/2)
    x2 = x1+patch.shape[0]
    y1 = position[1]-int(patch.shape[1]/2)
    y2 = y1+patch.shape[1]
    alpha = patch[:,:,3]
    alpha[alpha > 1] = 1
    segmentation_matrix[x1:x2,y1:y2] = alpha

def gaussian_alpha(patch):
    sigma = .8
    x, y = np.meshgrid(np.linspace(-1,1,patch.shape[1]), np.linspace(-1,1,patch.shape[0]))
    patch[:,:,3] *= np.exp(-(x*x+y*y)/(2.*sigma**2))

def filter_patch(patch, remove_labels=remove_labels_flag):
    """
    basic filtering based on unsharp mask and dark objects removal
    (they are detected by thresholding in Lab color-space, and replaced by the nearest valid pixel founded by random walk)
    """
    size_x = patch.shape[0]
    size_y = patch.shape[1]
    size_c = patch.shape[2]
    b = color.rgb2lab(patch[:,:,:3])[:,:,2] # b in Lab colorspace 
    # unsharp mask
    patch = unsharp_mask(patch, radius=100.0, amount=1.5, multichannel=True, preserve_range=False)
    # remove black areas
    L = color.rgb2lab(patch[:,:,:3])[:,:,0] # luminance in Lab colorspace
    #
    if remove_labels:
        if size_c == 3: patch[np.logical_and(L <= dark_matter_threshold, b < colonies_threshold)] = [1.,1.,1.]
        if size_c == 4: patch[np.logical_and(L <= dark_matter_threshold, b < colonies_threshold)] = [1.,1.,1.,0.]
        # 
        mask = np.zeros_like(patch[:,:,0], dtype=np.uint8)
        # detect dark regions via luminance and b-value thresholing
        mask[np.logical_and.reduce((L > dark_matter_threshold, L < dark_regions_threshold, b < colonies_threshold))] = 1 
        # dilate mask for better coverage of dark regions
        dilation_steps = int(np.sqrt(size_x*size_y)/64) # 36 # 128 # 256
        if dilation_steps > 16: dilation_steps = 16
        for i in range(dilation_steps): mask = dilation(mask) 
        where_mask = np.where(mask==1) 
        for i0,j0 in zip(where_mask[0],where_mask[1]):
            i = i0
            j = j0
            step = 2 # random walk starting step
            while mask[i,j]==1 and step < size_x and step < size_y: # random walk
                direction = int(rnd.random()*4)
                if direction == 0 and i < size_x-step: i += step
                elif direction == 1 and j < size_y-step: j += step
                elif direction == 2 and i > step-1: i -= step
                elif j > step-1: j -= step
                step += 2
            patch[i0,j0,:] = patch[i,j,:]
    return patch

def postpro_filtering(patch):
    """
    speckle noise cancellation
    """
    sigma_est = np.mean(estimate_sigma(patch, multichannel=True))
    patch_kw = dict(patch_size=5, patch_distance=13, multichannel=True)
    # denoising
    patch = denoise_nl_means(patch, h=1.5*sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    return patch

def segment_patch(patch):
    """
    patch segmentation by using robust Chan-Vese algorithm
    """
    i = np.arange(patch.shape[0])
    j = np.arange(patch.shape[1])
    ii, jj = np.meshgrid(j, i, sparse=True)
    init_set = (np.sin(ii/1*np.pi)*np.sin(jj/1*np.pi))**2
    return chan_vese(color.rgb2gray(patch[:,:,:3]), mu=0.5, lambda1=1, lambda2=1, 
                        tol=2e-3, max_iter=200,dt=0.5, init_level_set=init_set)

def get_alpha_from_segmentation(patch):
    """
    refine alpha matrix using segmentation
    """
    patch = filter_patch(patch)
    patch = postpro_filtering(patch)
    segmented = segment_patch(patch)
    alpha_matrix = np.zeros_like(segmented, dtype=np.uint8)
    alpha_matrix[segmented] = 1
    dilation_step = np.sqrt(patch.shape[0]**2+patch.shape[1]**2)/50.
    alpha_matrix = dilation(alpha_matrix, square(dilation_step.astype(np.uint16))) # adding margin to segmentation mask
    return alpha_matrix

def get_alpha_from_blending_with_backgroung(patch, alpha_from_seg, alpha_from_bboxes):
    """
    generate mask with lower values where pixel_color ~ patch_background_color: 
    to blend with empty dish backround
    """
    # patch in Lab colorspace
    patch_lab = color.rgb2lab(patch)
    # patch backround area
    background_colors = patch_lab[np.logical_and(alpha_from_bboxes == 1, alpha_from_seg == 0)]
    bgd_c = [np.mean(background_colors[:,i]) for i in range(3)] # average backround color
    # distance from backround color in Lab colorspace
    lab_dist = np.sqrt((patch_lab[:,:,0]-bgd_c[0])**2+(patch_lab[:,:,1]-bgd_c[1])**2+(patch_lab[:,:,2]-bgd_c[2])**2)  
    # normalization and weighting
    lab_dist /= np.amax(lab_dist)
    lab_dist = np.sqrt(np.sin(lab_dist*np.pi/2.))
    lab_dist = .6 + lab_dist*.4
    #
    return img_float2int(lab_dist, 255)

def segment_dish(dish, div=4):
    """
    segment subsequent patches
    """
    seg_matrix=np.zeros_like(dish[:,:,0], dtype=np.bool) 
    x_size = int(dish.shape[0]/div)
    x_rem = dish.shape[0] % x_size
    y_size = int(dish.shape[1]/div)
    y_rem = dish.shape[1] % y_size
    for i in range(div):
        for j in range(div):
            patch_x = x_size if i < div else x_size + x_rem
            patch_y = y_size if j < div else y_size + y_rem
            patch = dish[i*x_size:i*x_size+patch_x,j*y_size:j*y_size+patch_y,:]
            seg_matrix[i*x_size:i*x_size+patch_x,j*y_size:j*y_size+patch_y] = segment_patch(patch)
    return seg_matrix

def img_float2int(image, multiply = 1):
    return (image*multiply).astype(np.uint8)

def get_polygons(binary_mask):

    # Initialize variables
    obj = {}
    segmentation = []
    segmentation_polygons = []

    mask_list = np.ascontiguousarray(binary_mask)
    contours, hierarchy = cv2.findContours(mask_list, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    # Get the contours
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    # if len(segmentation) == 0:
    #     continue

    # Get the polygons as (x, y) coordinates
    for i in range(len(segmentation)):
        segment = segmentation[i]
        poligon = []
        poligons = []
        for j in range(len(segment)):
            poligon.append(segment[j])
            if (j + 1) % 2 == 0:
                poligons.append(poligon)
                poligon = []
        segmentation_polygons.append(poligons)

    # Save the segmentation and polygons for the current annotation
    return segmentation_polygons

def get_polygons_plot(segmentation_matrix):
    """
    return 2darray with the contours found
    """
    polygons = get_polygons(img_float2int(segmentation_matrix))
    polygons_plot = np.zeros_like(segmentation_matrix, dtype=np.uint8)
    for polygon in polygons:
        for point in polygon:
            polygons_plot[point[1],point[0]] = 1
    return polygons_plot

def transform_bboxes(json_data, patch, position, angle, mode):
    """
    transformation due to scalling, rotation and/or mirroring
    """
    px2 = patch.shape[0]/2
    py2 = patch.shape[1]/2
    pox = position[0]
    poy = position[1]
    for bbox in json_data:
        w = int(bbox['width']*scalling_factor) 
        h = int(bbox['height']*scalling_factor) 
        x = bbox['x']*scalling_factor
        y = bbox['y']*scalling_factor 
        if angle == 1:
            bw = w 
            bh = h
            bx = int( px2-x-w)
            by = int(-py2+y)           
        elif angle == 2:
            bw = h 
            bh = w
            bx = int( px2-y-h)
            by = int( py2-x-w)
        elif angle == 3:
            bw = w 
            bh = h
            bx = int(-px2+x)
            by = int( py2-y-h) 
        else:
            bw = h 
            bh = w
            bx = int(-px2+y)
            by = int(-py2+x) 
        if mode == 1: bx = -bx-bw
        if mode == 2: by = -by-bh
        bbox['width'] = bw
        bbox['height'] = bh
        bbox['x'] = int(pox+bx)
        bbox['y'] = int(poy+by) 

    return json_data

def isegmentation_mask(isegmentation_matrix, patch, position, patch_bboxes):
    colormap = cm.prism
    x1 = position[0]-int(patch.shape[0]/2)
    x2 = x1+patch.shape[0]
    y1 = position[1]-int(patch.shape[1]/2)
    y2 = y1+patch.shape[1]
    alpha = patch[:,:,3]
    alpha[alpha > 1] = 1
    no_clusters = len(patch_bboxes) 

    if no_clusters == 1:
        rgb = np.array(colormap(rnd.random()))[0:3]*rnd.randint(100,255)
        color_mask = np.stack((alpha*rgb[0],alpha*rgb[1],alpha*rgb[2])).transpose(1,2,0)
    else:
        to_clusterize = np.stack(np.where(alpha==1)).transpose()
        centroids = []
        for bbox in patch_bboxes: centroids.append([bbox['x']-x1 + int(bbox['width']/2),
                                                    bbox['y']-y1 + int(bbox['height']/2)])
        ctree = KDTree(np.array(centroids))
        labels = ctree.query(to_clusterize, p=2)[1]
        color_mask = np.zeros_like(patch[:,:,:3])
        for k in range(no_clusters):
            cluster = to_clusterize[np.where(labels==k)]
            rgb = np.array(colormap(rnd.random()))[0:3]*rnd.randint(100,255)
            for xy in cluster:
                color_mask[xy[0],xy[1],:] = rgb

    isegmentation_matrix[x1:x2,y1:y2,:] = color_mask

def bbox_dict(grouped_coordinates, classe):
    labels = []
    for coordinate in grouped_coordinates:
        coordinate.to_int()
        dicty = {}
        dicty["width"] = coordinate.x2-coordinate.x1
        dicty["height"] = coordinate.y2-coordinate.y1
        dicty["x"] = coordinate.x1
        dicty["y"] = coordinate.y1
        dicty["class"] = classe
        labels.append(dicty)
    out_dict = {}
    out_dict["labels"] = labels
    return out_dict

def bbox_plot(image_matrix, bbox_list):
    """
    plot bboxes on a given 2darray
    """
    for bbox in bbox_list:
        x1 = bbox['x']
        x2 = bbox['x'] + bbox['width']
        y1 = bbox['y']
        y2 = bbox['y'] + bbox['height']
        image_matrix[x1:x2,y1] = 1
        image_matrix[x2,y1:y2] = 1
        image_matrix[x1:x2,y2] = 1
        image_matrix[x1,y1:y2] = 1
    return image_matrix

def cut_from_dish(dish):
    """
    cut randomly recetangle of image_size x images_size from dish
    """
    x0 = rnd.randint(0,dish.shape[0]-image_size)
    y0 = rnd.randint(0,dish.shape[1]-image_size)
    dish = dish[x0:x0+image_size, y0:y0+image_size]
    return dish

def cut_patch(img, mask_coord, bbox_dict):
    bbox_mask = np.zeros_like(img, dtype=np.int8)
    labels = []
    for bbox in bbox_dict["labels"]:
        bbox_mask[:,:] = 0
        x1 = bbox['x']
        x2 = bbox['x'] + bbox['width']
        y1 = bbox['y']
        y2 = bbox['y'] + bbox['height']
        bbox_mask[x1:x2,y1:y2] = 1
        new_bbox_mask = bbox_mask[mask_coord[0]:mask_coord[1], mask_coord[2]:mask_coord[3]]
        if np.any(new_bbox_mask):
            nonzero_indices = np.stack(np.nonzero(new_bbox_mask),axis=1)
            x1 = np.amin(nonzero_indices[:,0])
            x2 = np.amax(nonzero_indices[:,0])
            y1 = np.amin(nonzero_indices[:,1])
            y2 = np.amax(nonzero_indices[:,1])
            if x2 > x1 and y2 > y1:
                dicty = {}
                dicty["width"] = int(x2-x1)
                dicty["height"] = int(y2-y1)
                dicty["x"] = int(x1)
                dicty["y"] = int(y1)
                dicty["class"] = bbox["class"]
                labels.append(dicty)
    new_bbox_dict = {}
    new_bbox_dict["labels"] = labels
    return img[mask_coord[0]:mask_coord[1], mask_coord[2]:mask_coord[3]], new_bbox_dict
