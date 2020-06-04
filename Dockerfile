FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN rm /etc/apt/sources.list.d/*  && rm -rf /var/lib/apt/lists/*

# Install Python 3.6
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update -y  && \
    apt-get install -y build-essential python3.6 python3.6-dev python3-pip && \
    apt-get autoremove && \
    apt-get clean

RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3
RUN pip3 install -U pip

# Install packages for inference
RUN pip3 install imageio matplotlib scikit-image easydict
RUN pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision==0.3.0

COPY models/correlation_package /root/correlation_package
RUN cd /root/correlation_package && python3 setup.py install
RUN mv /root/correlation_package /usr/local/lib/python3.6/dist-packages/

# Install dependencies for training
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get autoremove && \
    apt-get clean

RUN pip3 install 'opencv-python>=3.0,<4.0' path.py tensorboardX fast_slic

