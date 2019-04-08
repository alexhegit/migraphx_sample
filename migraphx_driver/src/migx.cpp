/*
 * migx - AMDMIGraphX driver program
 *
 * This driver program provides simple command line options for exercising
 * library calls associated with AMDMIGraphX graph library functions.
 * Included are options for the following stages of processing:
 *
 * 1. Loading saved models
 *
 * 2. Load input data used by the model
 *
 * 3. Quantize the model
 *
 * 4. Compile the program
 *
 * 5. Run program in various configurations
 *
 * More details about each of these options found with usage statement below.
 */
#include <iostream>
#include <iomanip>
#include <getopt.h>
#include <unistd.h>
#include <sys/time.h>
#include <migraphx/onnx.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/context.hpp>
#include "migx.hpp"
using namespace migraphx;
std::string migx_program; // argv[0] of this process
std::string usage_message =
  migx_program + " <options list>\n" + 
  "where <options list> includes options for\n" +
  "    general use\n" +
  "        --help\n" +
  "        --verbose\n" +
  "    loading saved models (either --onnx or --tfpb are required)\n" + 
  "        --onnx=<filename>\n" +
  "        --tfpb=<filename>\n" +
  "        --nhwc               set the data layout (--tfpb only)\n" +
  "        --nchw               set the data layout (default)\n" +
  "    quantization\n" +
  "        --fp16               quantize operations to float16\n" +
  "        --int8               quantize operations to int8\n" +
  "    input data\n" +
  "        --imagefile=<filename>\n"
  "    running\n" +
  "        --perf_report        run migraphx perf report including times per instruction\n" +
  "        --benchmark          run model repeatedly and time results\n" +
  "        --imageinfo          run model once and report top5 buckets for an image\n" +
  "        --imagenet=<dir>     run model on an imagenet directory\n" +
  "        --print_model        show MIGraphX instructions for model\n" +
  "        --iterations=<n>     set iterations for perf_report and benchmark (default 1000)\n" +
  "        --copyarg            copy arguments in and results back (--benchmark only)\n" +
  "        --argname=<name>     set name of model input argument (default 0)\n";


bool is_verbose = false;
enum model_type { model_unknown, model_onnx, model_tfpb } model_type = model_unknown;
std::string model_filename;
bool is_nchw = true;
enum quantize_type { quantize_none, quantize_fp16, quantize_int8 } quantize_type = quantize_none;
enum run_type { run_none, run_benchmark, run_perfreport, run_imageinfo, run_imagenet, run_printmodel } run_type = run_none;
int iterations = 1000;
bool copyarg = false;
std::string argname = "0";
std::string image_filename;
std::string imagenet_dir;

/* parse_options
 *
 * Parse user options, returning 0 on success
 */
int parse_options(int argc,char *const argv[]){
  int opt;
  static struct option long_options[] = {
    { "help",    no_argument,       0, 1 },
    { "verbose", no_argument,       0, 2 },
    { "onnx",    required_argument, 0, 3 },
    { "tfpb",    required_argument, 0, 4 },
    { "nhwc",    no_argument,       0, 5 },
    { "nchw",    no_argument,       0, 6 },
    { "fp16",    no_argument,       0, 7 },
    { "int8",    no_argument,       0, 8 },
    { "imagefile", required_argument, 0, 9 },
    { "benchmark", no_argument,     0, 10 },
    { "perf_report", no_argument,   0, 11 },
    { "imageinfo", no_argument,     0, 12 },
    { "imagenet", required_argument, 0, 13 },
    { "print_model", no_argument,   0, 14 },
    { "iterations", required_argument, 0, 15 },
    { "copyarg", no_argument,     0, 16 },
    { "argname", required_argument, 0, 17 },
  };
  while ((opt = getopt_long(argc,argv,"",long_options,NULL)) != -1){
    switch (opt){
    case 1:
      return 1;
    case 2:
      is_verbose = true;
      break;
    case 3:
      model_type = model_onnx;
      model_filename = optarg;
      break;
    case 4:
      model_type = model_tfpb;
      model_filename = optarg;
      break;
    case 5:
      is_nchw = false;
      break;
    case 6:
      is_nchw = true;
      break;
    case 7:
      quantize_type = quantize_fp16;
      break;
    case 8:
      quantize_type = quantize_int8;
      break;
    case 9:
      image_filename = optarg;
      break;
    case 10:
      run_type = run_benchmark;
      break;
    case 11:
      run_type = run_perfreport;
      break;
    case 12:
      run_type = run_imageinfo;
      break;
    case 13:
      imagenet_dir = optarg;
      run_type = run_imagenet;
    case 14:
      run_type = run_printmodel;
      break;
    case 15:
      if (std::stoi(optarg) < 0){
	std::cerr << migx_program << ": iterations < 0, ignored" << std::endl;
      } else {
	iterations = std::stoi(optarg);
      }
      break;
    case 16:
      copyarg = true;
      break;
    case 17:
      argname = optarg;
      break;
    default:
      return 1;
    }
  }
  if (model_type == model_unknown){
    std::cerr << migx_program << ": either --onnx or --tfpb must be given" << std::endl;
    return 1;
  }
  if (model_type == model_onnx && is_nchw == false){
    std::cerr << migx_program << ": --nhwc not supported with --onnx" << std::endl;
    return 1;
  }
  if ((run_type == run_imageinfo) && image_filename.empty()){
    std::cerr << migx_program << ": --imageinfo requires --imagefile option" << std::endl;
    return 1;
  }
  return 0;
}

/* get_time
 *
 * return current time in milliseconds
 */
double get_time(){
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return static_cast<double>(tv.tv_usec / 1000) + tv.tv_sec * 1000;
}

int main(int argc,char *const argv[],char *const envp[]){
  migx_program = argv[0];
  if (parse_options(argc,argv)){
    std::cerr << migx_program << ": usage: " << usage_message;
    return 1;
  }

  // load the model file
  if (is_verbose)
    std::cout << "loading model file" << std::endl;
  
  program prog;
  if (model_type == model_onnx){
    try {
      prog = parse_onnx(model_filename);
    } catch(...){
      std::cerr << migx_program << ": unable to load ONNX file " << model_filename << std::endl;
      return 1;
    }
  } else if (model_type == model_tfpb){
    try {
      prog = parse_tf(model_filename,!is_nchw);
    } catch(...){
      std::cerr << migx_program << ": unable to load TF protobuf file " << model_filename << std::endl;
      return 1;
    }
  }

  // quantize the program
  if (quantize_type != quantize_none){
    std::cerr << "quantization not yet implemented" << std::endl;
  }

  // compile the program
  if (is_verbose)
    std::cout << "compiling model" << std::endl;
  prog.compile(migraphx::gpu::target{});

  // set up the parameter map
  program::parameter_map pmap;
  std::cout << "allocating parameters" << std::endl;
  for (auto&& x: prog.get_parameter_shapes()){
    //    pmap[x.first] = migraphx::gpu::to_gpu(generate_argument(x.second));
    pmap[x.first] = migraphx::gpu::allocate_gpu(x.second);
  }
  enum image_type img_type;
  int batch_size = pmap[argname].get_shape().lens()[0];
  int channels = pmap[argname].get_shape().lens()[1]; // TODO: nhwc
  int height = pmap[argname].get_shape().lens()[2];   //
  int width = pmap[argname].get_shape().lens()[3];   //
  if (channels == 3 && height == 224 && width == 224) img_type = image_imagenet224;
  else if (channels == 3 && height == 299 && width == 299) img_type = image_imagenet299;
  else img_type = image_unknown;

  // read image data if passed
  std::vector<float> image_data(3*height*width);
  if (!image_filename.empty()){
    if (is_verbose)
      std::cout << "reading image: " << image_filename << " " << std::endl;
    read_image(image_filename,img_type,image_data);
  }

  migraphx::argument result;
  migraphx::argument resarg;
  double start_time,finish_time,elapsed_time;
  int top5[5];
  // alternatives for running the program
  auto ctx = prog.get_context();
  switch(run_type){
  case run_none:
    // do nothing
    break;
  case run_benchmark:
    int i;
    if (is_verbose && iterations > 1){
      std::cout << "running           " << iterations << " iterations" << std::endl;
    }
    start_time = get_time();
    for (i = 0;i < iterations;i++){
      if (copyarg)
	pmap[argname] = migraphx::gpu::to_gpu(generate_argument(prog.get_parameter_shape(argname)));
      resarg = prog.eval(pmap);
      ctx.finish();
      if (copyarg)
	result = migraphx::gpu::from_gpu(resarg);
    }
    finish_time = get_time();
    elapsed_time = (finish_time - start_time)/1000.0;

    std::cout << "batch size        " << batch_size << std::endl;        
    std::cout << std::setprecision(6) << "Elapsed time(ms): " << elapsed_time << std::endl;
    std::cout << "Images/sec:       " << (iterations*batch_size)/elapsed_time << std::endl;
    break;
  case run_perfreport:
    if (is_verbose && iterations > 1){
      std::cout << "running           " << iterations << " iterations" << std::endl;
    }
    prog.perf_report(std::cout,iterations,pmap);
    break;
  case run_imageinfo:
    pmap[argname] = migraphx::gpu::to_gpu(migraphx::argument{
	pmap[argname].get_shape(),image_data.data()});
    resarg = prog.eval(pmap);
    result = migraphx::gpu::from_gpu(resarg);
    image_top5((float *) result.data(), top5);
    std::cout << "top1 = " << top5[0] << " " << imagenet_labels[top5[0]] << std::endl;
    std::cout << "top2 = " << top5[1] << " " << imagenet_labels[top5[1]] << std::endl;
    std::cout << "top3 = " << top5[2] << " " << imagenet_labels[top5[2]] << std::endl;
    std::cout << "top4 = " << top5[3] << " " << imagenet_labels[top5[3]] << std::endl;
    std::cout << "top5 = " << top5[4] << " " << imagenet_labels[top5[4]] << std::endl;
    break;
  case run_imagenet:
    if (chdir(imagenet_dir) == -1){
      std::cerr << migx_program << ": can not change to imagenet dir: " << imagenet_dir << std::endl;
    }
    break;
  case run_printmodel:
    std::cout << prog;
    break;
  }
  return 0;
}
