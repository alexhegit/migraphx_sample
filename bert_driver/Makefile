MIGRAPHX_FOLDER=/home/mev/source/AMDMIGraphX
INCLUDES=-I $(MIGRAPHX_FOLDER)/src/include -I $(MIGRAPHX_FOLDER)/src/targets/gpu/include -I /opt/rocm/include
CXXFLAGS=
LIBS=-lmigraphx -lmigraphx_onnx -lmigraphx_tf -lmigraphx_device -lmigraphx_gpu -L $(MIGRAPHX_FOLDER)/build/src -L $(MIGRAPHX_FOLDER)/build/src/onnx -L $(MIGRAPHX_FOLDER)/build/src/tf -L $(MIGRAPHX_FOLDER)/build/src/targets/gpu -L /opt/rocm/lib
CXX=/opt/rocm/bin/hcc

bert_mrpc:	bert_mrpc.cpp
	$(CXX) -g -o bert_mrpc $(CXXFLAGS) $(INCLUDES) bert_mrpc.cpp $(LIBS)

clean:
	-rm *.o *~

clobber:	clean
	-rm bert_mrpc
