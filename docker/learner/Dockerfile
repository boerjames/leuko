# TRAINER IMAGE
FROM centos:7

# COMMON DEPENDENCIES
RUN yum install -y gcc vim sudo wget 

# CUDA DEPENDENCIES
RUN yum install -y epel-release

# CUDA
RUN yum install -y http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-7.5-18.x86_64.rpm
RUN yum install -y nvidia-kmod
RUN yum install -y nvidia-uvm-kmod
RUN yum install -y cuda
RUN yum install -y cuda-drivers
RUN yum install -y cuda-nvidia-kmod-common
RUN yum install -y cuda*7-5
RUN yum clean all
RUN yum update -y

# CUDNN
COPY cudnn-7.0-linux-x64-v3.0-prod.tgz /tmp

RUN cd /tmp && \
	tar -zxvf cudnn-7.0-linux-x64-v3.0-prod.tgz && \
	cp cuda/include/* /usr/local/cuda-7.5/include/ && \
	cp cuda/lib64/* /usr/local/cuda-7.5/lib64/

# CUDA ENVIRONMENT
RUN touch /etc/profile.d/cuda.sh
RUN echo -e "export PATH=/usr/local/cuda-7.5/bin:$PATH" > /etc/profile.d/cuda.sh
RUN echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH" >> /etc/profile.d/cuda.sh

# WORKING DIRECTORY
WORKDIR /root

# EDIT SUDO FILE: DO NOT WANT TO REQUIRE TTY FOR SUDO
RUN sed -i -e "s/Defaults    requiretty.*/ #Defaults    requiretty/g" /etc/sudoers

# TORCH
RUN curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch; ./install.sh

# TORCH ENVIRONMENT
ENV LUA_PATH='/ioot/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-alpha/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
  LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
  PATH=/root/torch/install/bin:$PATH \
  LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
  DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH

# TORCH LIBRARIES
RUN luarocks install dp
RUN luarocks install cutorch
RUN luarocks install cunn
RUN cd ~ && git clone https://github.com/soumith/cudnn.torch.git && \
	cd cudnn.torch && git checkout R3 && luarocks make

# DEEP ENVIRONMENT
ENV DEEP_DATA_PATH=/root/deep/data \
	DEEP_SAVE_PATH=/root/deep/save \
	DEEP_UNIT_PATH=/root/deep/unit \
	DEEP_LOG_PATH=/root/deep/log
