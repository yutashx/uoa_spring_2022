FROM nvidia/cuda:11.0-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip neovim tree lsof
RUN pip3 install torch torchvision pandas pillow

WORKDIR /work

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
