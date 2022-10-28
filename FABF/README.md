
## Fast Adaptive Bilateral Filtering

This is a Matlab implementation of the algorithm in the following paper:

R. G. Gavaskar and K. N. Chaudhury, "Fast Adaptive Bilateral Filtering", IEEE Transactions on Image Processing, vol. 28, no. 2, pp. 779-790, 2019.

DOI: 10.1109/TIP.2018.2871597

[[IEEEXplore]](https://ieeexplore.ieee.org/document/8469064) [[arXiv]](https://arxiv.org/abs/1811.02308)

### Requirements

(1) Matlab with Image Processing Toolbox.

(2) C++ compiler (to compile mex file).

Tested on Matlab 9.1.0 (R2016b) and GCC 4.8.4 (Ubuntu 14.04).

### Details

Before running the code, compile the MEX file in the 'fastABF' directory as follows:
```
mex MinMaxFilter.cpp
```
This is the O(1) filter to find local windowed minima and maxima (alpha and beta in the paper).

The core source files implementing the algorithm are located in 'fastABF'.
To execute the algorithm, run the command
```
g = fastABF( f,rho,sigma,theta )
```
where
```
f       = input image (m-by-n),
rho     = width of the spatial Gaussian kernel,
sigma   = width of the range Gaussian kernel, defined pixelwise (m-by-n),
theta   = centering of the range Gaussian kernel, defined pixelwise (m-by-n).
```

Note that the input image f is assumed to be in the range [0,255] (or close to this range, for noisy images). The values of sigma_r must be set keeping this range in mind. The code may not work for images taking values in [0,1].

The main directory contains files to demonstrate application of the algorithm for image sharpening and noise removal, texture filtering, and JPEG deblocking.
Run the files 'demo_sharpening.m', 'demo_texture.m', and 'demo_deblocking.m' respectively.

### Citation
```
@article{Gavaskar_Chaudhury_2019,
  author = {R. G. Gavaskar and K. N. Chaudhury}, 
  journal = {IEEE Transactions on Image Processing}, 
  title = {Fast Adaptive Bilateral Filtering}, 
  year = {2019}, 
  volume = {28}, 
  number = {2}, 
  pages = {779--790}, 
  month = {Feb},
}
```

### Credits

The image 'fish.jpg' used for texture filtering has been downloaded from the [project webpage](http://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/index.html) for the following paper:

L. Xu, Q. Yan, Y. Xia, and J. Jia, ''Structure extraction from texture via relative total variation,'' ACM Transactions on Graphics (TOG), 31(6), Article 139 (2012).

### Download

An up-to-date version of this software is available here: https://github.com/rgavaska/Fast-Adaptive-Bilateral-Filtering.
