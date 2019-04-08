/*
 * image.cpp - image processing using OpenCV
 */
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include "migx.hpp"
using namespace cv;

void read_image(std::string filename,enum image_type etype,std::vector<float> &image_data){
  Mat img,scaleimg,cropimg;
  int resize_num, image_size;
  
  switch(etype){
  case image_unknown:
    std::cerr << migx_program << ": unknown image type" << std::endl;
    return;
  case image_imagenet224:
    resize_num = 256;
    image_size = 224;
    break;
  case image_imagenet299:
    resize_num = 299;
    image_size = 299;
    break;
  default:
    std::cout << "imagenet ??? " << etype << std::endl;
    break;
  }
  
  img = imread(filename,CV_LOAD_IMAGE_COLOR);
  if (!img.data){
    std::cerr << migx_program << ": unable to load image file " << filename << std::endl;
    return;
  }
  // resize the image for imagenet
  double scale = resize_num / (double) min(img.rows,img.cols);
  resize(img,scaleimg,Size(img.cols*scale,img.rows*scale));
  // center crop to appropriate size
  cropimg = scaleimg(Rect((scaleimg.cols-image_size)/2,
			  (scaleimg.rows-image_size)/2,
			  image_size,image_size));
  // TODO: normalize to: mean[0.485,0.456,0.406] std[0.229,0.224,0.224]
  // Change from HWC to CHW and convert to float32
  Mat_<Vec3b> _image = cropimg;
  for (int i=0;i < image_size;i++)
    for (int j=0;j < image_size;j++){
      image_data[0*image_size*image_size + i*image_size + j] = _image(i,j)[0]/256.0;
      image_data[1*image_size*image_size + i*image_size + j] = _image(i,j)[1]/256.0;
      image_data[2*image_size*image_size + i*image_size + j] = _image(i,j)[2]/256.0;
    }     
}

// return the indices of the top elements, simple iterative algorithm
// quick and dirty implementation, I expect there is a std::sort approach as well...
struct float_idx { float value; int index; };
int compare_float_idx(const void *elt1,const void *elt2){
  if (*((float *) elt1) > *((float *) elt2)) return 1;
  else if (*((float *) elt1) < *((float *) elt2)) return -1;
  else return 0;
}

void image_top5(float* array,int *top5){
  int i;
  struct float_idx sort_array[1000];
  for (int i=0;i<1000;i++){
    sort_array[i].value = array[i];
    sort_array[i].index = i;
  }
  qsort(sort_array,1000,sizeof(struct float_idx),compare_float_idx);
  top5[0] = sort_array[999].index;
  top5[1] = sort_array[998].index;
  top5[2] = sort_array[997].index;
  top5[3] = sort_array[996].index;
  top5[4] = sort_array[995].index;
}
