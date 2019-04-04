# Makefile for building "migx", migraphx driver program
#
CXX=g++
CFLAGS=-g

migx:	migx.o
	$(CXX) -o migx migx.o

migx.o:	migx.cpp
	$(CXX) -c $(CFLAGS) migx.cpp

clean:
	-rm *.o *~

clobber:	clean
	-rm migx
