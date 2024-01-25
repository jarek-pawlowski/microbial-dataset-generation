## Generation of microbial colonies dataset 

This is repository with the code used to generate synthetic microbiological dataset in the paper "Generation of microbial colonies dataset with deep learning style transfer" [1]. During generation we use images from AGAR [2] -- recently introduced large microbiological dataset.

#### Prerequisites:
- Before synthesis one has to download the input data. We used 100 images from the _higher-resolution_ AGAR subset -- it can be downloaded [here](https://api.data.neurosys.com:4443/agar-public/microbial_dataset_generation.zip) together with the images containing empty dishes (on which colonies will be placed) + images that carries style (used to style transfer).
- Then, set up environment (we recommend conda, and python 3.7) -- see requirements.txt.

#### Usage:
The process consists of two parts: (1) extracting colonies from real images, and (2) generate synthetic images using the extracted colonies and apply style transfer.
- to extact microbial colonies from the input images of Petri dish:
```bash
python get_patches.py -i ./input_data -o ./colonies
-i  : directory containing input images
-o  : ditectory to store extracted colonies
```
- to generate synthetic patches -- fragments of dish with placed (previously extracted) colonies and apply style transfer:
```bash
python grow_colonies.py -c ./colonies -e ./empty_dishes -s ./style_dishes -o ./generated
-c  : directory containing colonies that will be used during generation
-e  : directory containing images of empty dishes used during generation
-s  : directory containing images carrying style to be transferred during the stylization stage
-o  : ditectory to store generated patches
```

#### Notes:
- labels for the each generated patch are stored in *.json file in the COCO format,
- instance segmentation masks are stored in .npy files,
- it is important to use scikit-image in version 0.17.2,
- neural style transfer part is based on the repository provided in [3],
- if you find this repository useful, please cite us :)

#### Literature:
[1] J. Pawłowski, S. Majchrowska, & T. Golan, Generation of microbial colonies dataset with deep learning style transfer, _Scientific Reports_ 12, 5212 (2022).\
[2] S. Majchrowska et al., AGAR a microbial colony dataset for deep learning detection, arXiv preprint: 2108.01234 (2021).\
[3] M. Li, C. Ye, & W. Li, High-resolution network for photorealistic style transfer, arXiv preprint: 1904.11617 (2019).

