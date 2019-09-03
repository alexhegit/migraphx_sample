#include <iostream>
#include <string>
#include <vector>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>

// hard coded values of tokenized inputs for 1st GLUE MRPC entry
std::vector<int64_t> input_ids{ 101, 1124, 1163, 1103, 11785, 1200, 14301, 16288, 1671, 2144, 112, 189, 4218, 1103, 1419, 112, 188, 1263, 118, 1858, 3213, 5564, 119, 102, 107, 1109, 11785, 1200, 14301, 16288, 1671, 1674, 1136, 4218, 1412, 1263, 118, 1858, 3213, 5564, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
std::vector<int64_t> input_mask{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
std::vector<int64_t> sequence_ids{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

void mrpc_test_onnx(std::string filename){
  // load ONNX file
  auto prog = migraphx::parse_onnx(filename);
  std::cout << prog << std::endl;

  // compile
  prog.compile(migraphx::gpu::target{});

  // pass in arguments
  for (auto&& x: prog.get_parameter_shapes()){
    std::cout << "parameter: " << x.first << " shape: " << x.second << std::endl;
  }

  migraphx::program::parameter_map pmap;
  pmap["scratch"] = migraphx::gpu::allocate_gpu(prog.get_parameter_shape("scratch"));
  pmap["output"] = migraphx::gpu::allocate_gpu(prog.get_parameter_shape("output"));

  migraphx::argument arg{};
  arg = migraphx::argument(prog.get_parameter_shape("input.1"),input_ids.data());
  pmap["input.1"] = migraphx::gpu::to_gpu(arg);
  arg = migraphx::argument(prog.get_parameter_shape("2"),input_mask.data());
  pmap["2"] = migraphx::gpu::to_gpu(arg);    
  arg = migraphx::argument(prog.get_parameter_shape("input.3"),sequence_ids.data());
  pmap["input.3"] = migraphx::gpu::to_gpu(arg);

  // evaluate
  auto result = migraphx::gpu::from_gpu(prog.eval(pmap));
  std::vector<float> vec_output;
  result.visit([&](auto output){ vec_output.assign(output.begin(),output.end()); });
  std::cout << "result = " << vec_output[0] << ", " << vec_output[1] << std::endl;
}

void mrpc_test_tf(std::string filename){
  // load TF file
  auto prog = migraphx::parse_tf(filename,true);
  std::cout << prog << std::endl;

  // compile
  prog.compile(migraphx::gpu::target{});

  // pass in arguments
  for (auto&& x: prog.get_parameter_shapes()){
    std::cout << "parameter: " << x.first << " shape: " << x.second << std::endl;
  }
  std::vector<int32_t> input_ids32(input_ids.begin(),input_ids.end());
  std::vector<int32_t> input_mask32(input_mask.begin(),input_mask.end());
  std::vector<int32_t> sequence_ids32(sequence_ids.begin(),sequence_ids.end());

  migraphx::program::parameter_map pmap;
  pmap["scratch"] = migraphx::gpu::allocate_gpu(prog.get_parameter_shape("scratch"));
  pmap["output"] = migraphx::gpu::allocate_gpu(prog.get_parameter_shape("output"));

  migraphx::argument arg{};
  arg = migraphx::argument(prog.get_parameter_shape("input_ids_1"),input_ids32.data());
  pmap["input_ids_1"] = migraphx::gpu::to_gpu(arg);
  arg = migraphx::argument(prog.get_parameter_shape("input_mask_1"),input_mask32.data());
  pmap["input_mask_1"] = migraphx::gpu::to_gpu(arg);
  arg = migraphx::argument(prog.get_parameter_shape("segment_ids_1"),sequence_ids32.data());
  pmap["segment_ids_1"] = migraphx::gpu::to_gpu(arg);      

  // evaluate
  auto result = migraphx::gpu::from_gpu(prog.eval(pmap));
  std::vector<float> vec_output;
  result.visit([&](auto output){ vec_output.assign(output.begin(),output.end()); });
  std::cout << "result = " << vec_output[0] << ", " << vec_output[1] << std::endl;  
}

int main(int argc, char **argv){
  if (argc != 3){
    std::cout << "Usage: " << argv[0] << " onnx|tf filename" << std::endl;
    return 0;
  }
  if (std::string(argv[1]) == "onnx")
    mrpc_test_onnx(argv[2]);
  else if (std::string(argv[1]) == "tf")
    mrpc_test_tf(argv[2]);
}
