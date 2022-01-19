import os
import copy
import json
import click

import numpy as np
import random as rnd
import datetime as dt

from skimage import io, color
import skimage.transform as tr

import lib
import transfer_style_lib

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-c', '--colonies_dir', help='path to directory containing previously extracted colonies')
@click.option('-e', '--empty_dishes_dir', help='path to directory containing previously extracted colonies')
@click.option('-s', '--style_dir', help='path to directory containing previously extracted colonies')
@click.option('-o', '--generated_dir', help='path to directory that will store generated patches')
def main(colonies_dir, empty_dishes_dir, style_dir, generated_dir):

    patches_dirs = {key:os.path.join(colonies_dir, value) for key, value in lib.patches_dirs.items()}
    empty_dishes = empty_dishes_dir
    style_dirs = {key:os.path.join(style_dir, value) for key, value in lib.style_dirs.items()}
    generated_dishes = generated_dir
    
    if not os.path.exists(generated_dishes): os.mkdir(generated_dishes)

    bacteria_flavours = [key for (key,value) in patches_dirs.items()]
    no_of_bacteria_flavours = len(patches_dirs)
    patches_files_dict = {}
    for (key,value) in patches_dirs.items():
	    listdir = os.listdir(value)
	    listdir = [f for f in listdir if f.endswith('.png')]
	    patches_files_dict[key] = listdir
    empty_files = os.listdir(empty_dishes)
    empty_no = len(empty_files)
    style_files_dict = {key:os.listdir(value) for (key,value) in style_dirs.items()}

    style_net_model = None

    rnd.seed()

    while True:
	    # randomize bacteria type
	    flavour = bacteria_flavours[rnd.randrange(no_of_bacteria_flavours)]
	    patches_files = patches_files_dict[flavour]
	    patches_no = len(patches_files)
	    styles_files = style_files_dict[flavour]
	    styles_no = len(styles_files)
	    # randimize empty dish
	    dish_path = os.path.join(empty_dishes, empty_files[rnd.randrange(empty_no)]) 
	    dish = io.imread(dish_path)
	    dish = tr.rotate(dish, rnd.random()*360, resize=False, preserve_range=True)
	    dish = lib.cut_from_dish(dish)  # cut empty patch from an empty dish 
	    #dish = np.dstack((dish, np.ones((dish.shape[0],dish.shape[1]), dtype=np.uint8)*255)) # add alpha channel
	    occupation_matrix = np.zeros((dish.shape[0],dish.shape[1]), dtype=np.uint8)
	    segmentation_matrix = np.zeros((dish.shape[0],dish.shape[1]), dtype=np.uint8)
	    isegmentation_matrix = np.zeros((dish.shape[0],dish.shape[1],3), dtype=np.uint8)
	    # exp-like distribution for no of bacteria colonies per dish 
	    lambd=0.07 # like in >>distr.txt<< file
	    colonies_no = int(rnd.expovariate(lambd))+1
	    
	    bboxes_labels = []
	    for i in range(colonies_no):
		    ir = rnd.randrange(patches_no)
		    patch_path = os.path.join(patches_dirs[flavour], patches_files[ir])
		    patch = io.imread(patch_path)
		    patch = tr.rescale(patch, lib.scalling_factor, anti_aliasing=True, multichannel=True)*255
		    # increase variety of patches
		    patch_rotation_angle = rnd.randint(0,3)
		    patch = tr.rotate(patch, patch_rotation_angle*90, resize=True)
		    flip_mode = rnd.randint(0,2)
		    if flip_mode > 0:
			    if flip_mode == 1: patch = patch[::-1,:,:]
			    if flip_mode == 2: patch = patch[:,::-1,:]
		    #
		    #patch = np.dstack((patch, np.ones((patch.shape[0],patch.shape[1]), dtype=np.uint8)*255)) # add alpha channel
		    lib.gaussian_alpha(patch)
		    try:
			    position = lib.rand_position(occupation_matrix, patch.shape[:2]) # randomize colony position
		    except Exception as e:
			    print(repr(e))
			    continue
		    lib.blend_patch(occupation_matrix, dish, patch, position)
		    lib.segmentation_mask(segmentation_matrix, patch, position)
		    # load bboxes for the given patch
		    json_path = patch_path[:-3] + "json"
		    with open(json_path, 'r') as jf:
			    json_data = json.load(jf)
			    patch_bboxes = lib.transform_bboxes(json_data['labels'], patch, position, patch_rotation_angle, flip_mode)
			    bboxes_labels.extend(patch_bboxes)
			    lib.isegmentation_mask(isegmentation_matrix, patch, position, patch_bboxes)

	    # apply style-transfer
	    style_path = os.path.join(style_dirs[flavour], styles_files[rnd.randrange(styles_no)]) 
	    style_image = io.imread(style_path)
	    dish, style_net_model = transfer_style_lib.style_transfer(lib.img_float2int(dish), style_image, style_net_model)

	    gen_name  = dt.datetime.now().strftime("%y%m%d_%H%M%S") 
	    bbox_dict = {}
	    bbox_dict["labels"] = bboxes_labels

	    suffix = "_1_"
	    cut_patch, new_bbox_dict = lib.cut_patch(dish, [0, int(dish.shape[0]/2), 0, int(dish.shape[1]/2)], bbox_dict)
	    cut_iseg = isegmentation_matrix[0: int(dish.shape[0]/2), 0: int(dish.shape[1]/2)]
	    #
	    cut_isegu = cut_iseg[:,:,0]*65536 + cut_iseg[:,:,1]*256 + cut_iseg[:,:,2]
	    u, indices = np.unique(cut_isegu.flatten(), return_inverse=True)
	    if len(u) > 1:
		    cut_isegu = np.reshape(indices, (512,512)).astype(np.uint8)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + "_iseg.png"), lib.img_float2int(cut_iseg, 255))
		    np.save(os.path.join(generated_dishes, gen_name + suffix), cut_isegu)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + ".png"), lib.img_float2int(cut_patch, 255))
		    with open(os.path.join(generated_dishes, gen_name + suffix + ".json"), 'w') as jf: json.dump(new_bbox_dict, jf)

	    suffix = "_2_"
	    cut_patch, new_bbox_dict = lib.cut_patch(dish, [0, int(dish.shape[0]/2), int(dish.shape[1]/2), dish.shape[1]], bbox_dict)
	    cut_iseg = isegmentation_matrix[0: int(dish.shape[0]/2), int(dish.shape[1]/2): dish.shape[1]]
	    #
	    cut_isegu = cut_iseg[:,:,0]*65536 + cut_iseg[:,:,1]*256 + cut_iseg[:,:,2]
	    u, indices = np.unique(cut_isegu.flatten(), return_inverse=True)
	    if len(u) > 1:
		    cut_isegu = np.reshape(indices, (512,512)).astype(np.uint8)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + "_iseg.png"), lib.img_float2int(cut_iseg, 255))
		    np.save(os.path.join(generated_dishes, gen_name + suffix), cut_isegu)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + ".png"), lib.img_float2int(cut_patch, 255))
		    with open(os.path.join(generated_dishes, gen_name + suffix + ".json"), 'w') as jf: json.dump(new_bbox_dict, jf)

	    suffix = "_3_"
	    cut_patch, new_bbox_dict = lib.cut_patch(dish, [int(dish.shape[0]/2), dish.shape[0], 0, int(dish.shape[1]/2)], bbox_dict)
	    cut_iseg = isegmentation_matrix[int(dish.shape[0]/2): dish.shape[0], 0: int(dish.shape[1]/2)]
	    #
	    cut_isegu = cut_iseg[:,:,0]*65536 + cut_iseg[:,:,1]*256 + cut_iseg[:,:,2]
	    u, indices = np.unique(cut_isegu.flatten(), return_inverse=True)
	    if len(u) > 1:
		    cut_isegu = np.reshape(indices, (512,512)).astype(np.uint8)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + "_iseg.png"), lib.img_float2int(cut_iseg, 255))
		    np.save(os.path.join(generated_dishes, gen_name + suffix), cut_isegu)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + ".png"), lib.img_float2int(cut_patch, 255))
		    with open(os.path.join(generated_dishes, gen_name + suffix + ".json"), 'w') as jf: json.dump(new_bbox_dict, jf)
	    
	    suffix = "_4_"
	    cut_patch, new_bbox_dict = lib.cut_patch(dish, [int(dish.shape[0]/2), dish.shape[0], int(dish.shape[1]/2), dish.shape[1]], bbox_dict)
	    cut_iseg = isegmentation_matrix[int(dish.shape[0]/2): dish.shape[0], int(dish.shape[1]/2): dish.shape[1]]
	    #
	    cut_isegu = cut_iseg[:,:,0]*65536 + cut_iseg[:,:,1]*256 + cut_iseg[:,:,2]
	    u, indices = np.unique(cut_isegu.flatten(), return_inverse=True)
	    if len(u) > 1:
		    cut_isegu = np.reshape(indices, (512,512)).astype(np.uint8)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + "_iseg.png"), lib.img_float2int(cut_iseg, 255))
		    np.save(os.path.join(generated_dishes, gen_name + suffix), cut_isegu)
		    io.imsave(os.path.join(generated_dishes, gen_name + suffix + ".png"), lib.img_float2int(cut_patch, 255))
		    with open(os.path.join(generated_dishes, gen_name + suffix + ".json"), 'w') as jf: json.dump(new_bbox_dict, jf)

if __name__ == '__main__':
    main()
