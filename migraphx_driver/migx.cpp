/*
 * migx - AMDMIGraphX driver program
 *
 * This driver program provides simple command line options for exercising
 * library calls associated with AMDMIGraphX graph library functions.
 * Included are options for the following stages of processing:
 *
 * 1. Loading saved models
 *    --onnx=<filename>         load from a saved ONNX file
 *    --tfpb=<filename>         load from a saved TF protobuf file
 *        --nhwc                data format nhwc (--tfpb only)
 *        --nchw                data format nchw (--tfpb only)
 *
 * 2. Load input data used by the model
 *    --randominput             default option if nothing is provided
 *    --image=<filename>        image file
 *
 * 3. Quantize the model
 *    --fp16
 *    --int8
 *
 * 4. Compile the program
 *
 * 5. Run program in various configurations
 *    --benchmark               run model for n interations (default 1000)
 *       --iterations
 *    --perf-report             predefined performance report
 *    --imagenet=<directory>    run imagenet validation tests
 */
