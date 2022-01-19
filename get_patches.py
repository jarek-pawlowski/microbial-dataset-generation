import os
import json
import click

import numpy as np
import networkx as nx
import skimage.io as io

import lib


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-i', '--input_dir', help='path to directory containing input images')
@click.option('-o', '--output_dir', help='path to directory that will store extracted colonies')
def main(input_dir, output_dir):

    dataset_dir = input_dir
    patches_dir = output_dir
    if not os.path.exists(patches_dir): os.mkdir(patches_dir)

    for img in os.listdir(dataset_dir):
        if img.endswith(".jpg"):
            img_path = os.path.join(dataset_dir, img)
            img_name = os.path.splitext(img)[0]
            scale = 1.
            if len(img_name) > 5:
                img_name = img_name[:-8]
                scale = lib.translate_factor
            json_path = os.path.join(dataset_dir, img_name + ".json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    labels = data['labels']
                    species = ["0", "1", "2", "3", "4", "6"] #data['bacterias']
                    for specie in species:
                        specie_patches_dir = os.path.join(patches_dir, specie)
                        if not os.path.exists(specie_patches_dir): os.mkdir(specie_patches_dir)
                    coordinates = []
                    classes = []
                    alphas = {}
                    for label in labels:
                        coordinates.append(lib.Coordinate(np.array([label['x'],label['x']+label['width'],\
                                                                label['y'],label['y']+label['height']])))
                        clas = label['class']
                        if not clas.isnumeric(): clas = lib.category_id_dict[clas] 
                        classes.append(clas)
                    coordinates = [c*scale for c in coordinates]
                    for c in coordinates: c.to_int()
                    petri_image = io.imread(img_path)
                    print(img_path)
                    # extend patches in case of colonies overlapping
                    adj_matrix = lib.adjacency_matrix_between_patches(coordinates)
                    G = nx.from_numpy_matrix(adj_matrix) 
                    connected_patches_list = list(nx.connected_components(G))
                    patches_to_del = []
                    patches_grouped_coordinates = {}
                    for connected_patches in connected_patches_list:
                        connected_patches = list(connected_patches)
                        print(connected_patches)
                        if len(connected_patches) > 1: # overlapping
                            # coordinates of extended patch
                            ex1 = np.amin([coordinates[i].x1 for i in connected_patches])
                            ex2 = np.amax([coordinates[i].x2 for i in connected_patches])
                            ey1 = np.amin([coordinates[i].y1 for i in connected_patches])
                            ey2 = np.amax([coordinates[i].y2 for i in connected_patches])
                            # alpha channel for non-rectangular cropping of extended patch
                            alpha_matrix = np.zeros((ex2-ex1,ey2-ey1), dtype=np.uint8)
                            grouped_coordinates = []
                            for i in connected_patches:
                                alpha_matrix[coordinates[i].x1-ex1:coordinates[i].x2-ex1,
                                                coordinates[i].y1-ey1:coordinates[i].y2-ey1] = 1
                                grouped_coordinates.append(lib.Coordinate([coordinates[i].x1-ex1,\
                                                                            coordinates[i].x2-ex1,\
                                                                            coordinates[i].y1-ey1,\
                                                                            coordinates[i].y2-ey1]))
                            alphas[connected_patches[0]] = alpha_matrix
                            patches_grouped_coordinates[connected_patches[0]] = grouped_coordinates
                            # extended patch
                            coordinates[connected_patches[0]] = lib.Coordinate([ex1,ex2,ey1,ey2])
                            patches_to_del.extend(connected_patches[1:])
                        else:
                            c = coordinates[connected_patches[0]]
                            alphas[connected_patches[0]] = np.ones((c.x2-c.x1,c.y2-c.y1), dtype=np.uint8)
                            patches_grouped_coordinates[connected_patches[0]] = [lib.Coordinate([0,c.x2-c.x1,0,c.y2-c.y1])]

                    coordinates = lib.del_multiple(coordinates, patches_to_del)
                    classes = lib.del_multiple(classes, patches_to_del)
                    alphas = [alphas[a] for a in sorted(alphas)] # to arrange alpha matrices in the same order as coordinates
                    patches_grouped_coordinates = [patches_grouped_coordinates[c] for c in sorted(patches_grouped_coordinates)]

                    for ic, (classe,coordinate) in enumerate(zip(classes,coordinates)):
                        patch_image = petri_image[coordinate.y1:coordinate.y2,coordinate.x1:coordinate.x2]
                        # join alphas                    
                        alpha_from_bboxes = alphas[ic].transpose()
                        try:
                            alpha_from_seg = lib.get_alpha_from_segmentation(patch_image)
                            alpha_from_blending = lib.get_alpha_from_blending_with_backgroung(patch_image, alpha_from_seg, alpha_from_bboxes)
                        except ValueError: 
                            print("ValueError")
                            continue
                        # join three alpha-matrices
                        alpha_matrix = np.multiply(alpha_from_seg, alpha_from_bboxes)
                        alpha_matrix = np.multiply(alpha_from_blending, alpha_matrix)
                        #
                        patch_image = np.dstack((patch_image, alpha_matrix)) # add alpha channel
                        specie_patch_dir = os.path.join(patches_dir, classe)
                        io.imsave(os.path.join(specie_patch_dir, img_name + "_" + str(ic) + ".png"), patch_image)
                        # write json with proper bboxes     
                        with open(os.path.join(specie_patch_dir, img_name + "_" + str(ic) + ".json"), 'w') as jf:
                            json.dump(lib.bbox_dict(patches_grouped_coordinates[ic],classe), jf)

if __name__ == '__main__':
    main()  
