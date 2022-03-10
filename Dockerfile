FROM nvidia/cuda:11.0-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip neovim tree lsof
RUN pip3 install pandas pillow
RUN pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

WORKDIR /work

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
