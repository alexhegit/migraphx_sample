// definitions for migx driver

extern std::string migx_program;
extern const char *imagenet_labels[];
enum image_type { image_unknown=0, image_imagenet224, image_imagenet299 };

void read_image(std::string filename,enum image_type etype,std::vector<float> &img_data);
void image_top5(float *res,int *top5);
