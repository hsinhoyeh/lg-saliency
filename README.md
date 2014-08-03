Salient Object Detection via Local Saliency Estimation and Global Homogeneity Refinement
=========================

This matlab demonstrates a simple implementation of detecting saliency map from an image.

How to use:

1. define inputs: a cell array, where each element contains the filepath of image

```
files = { 'a/b/c/d.jpg', 'a/b/c/e.jpg'}
```

2. call rare2comp_icip functions, the salient map of each images given is returned.
```
salmap = local2global_object_saliency_superpatch8(files)
```


How to cite this paper:

Hsin-Ho Yeh, Keng-Hao Liu, and Chu-Song Chen, "Salient Object Detection via Local Saliency Estimation and Global Homogeneity Refinement," Pattern Recognition, volume 47, number 4, pages 1740–1750, April 2014.
