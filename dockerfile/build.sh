#!/bin/bash
env DOCKERFILE=Dockerfile-16.04:2.3 ./build_dockerfile.sh <<EOF
2.3
16.04
EOF
env DOCKERFILE=Dockerfile-16.04:2.4 ./build_dockerfile.sh <<EOF
2.4
16.04
EOF
env DOCKERFILE=Dockerfile-16.04:2.5 ./build_dockerfile.sh <<EOF
2.5
16.04
EOF
env DOCKERFILE=Dockerfile-18.04:2.3 ./build_dockerfile.sh <<EOF
2.3
18.04
EOF
env DOCKERFILE=Dockerfile-18.04:2.4 ./build_dockerfile.sh <<EOF
2.4
18.04
EOF
env DOCKERFILE=Dockerfile-18.04:2.5 ./build_dockerfile.sh <<EOF
2.5
18.04
EOF
docker build -f Dockerfile-16.04:2.3 -t migraphx-sample-16.04:2.3 . 2>&1 | tee dockerbuild-16.04:2.3.log
docker build -f Dockerfile-16.04:2.4 -t migraphx-sample-16.04:2.4 . 2>&1 | tee dockerbuild-16.04:2.4.log
docker build -f Dockerfile-16.04:2.5 -t migraphx-sample-16.04:2.5 . 2>&1 | tee dockerbuild-16.04:2.5.log
docker build -f Dockerfile-18.04:2.3 -t migraphx-sample-18.04:2.3 . 2>&1 | tee dockerbuild-18.04:2.3.log
docker build -f Dockerfile-18.04:2.4 -t migraphx-sample-18.04:2.4 . 2>&1 | tee dockerbuild-18.04:2.4.log
docker build -f Dockerfile-18.04:2.5 -t migraphx-sample-18.04:2.5 . 2>&1 | tee dockerbuild-18.04:2.5.log
