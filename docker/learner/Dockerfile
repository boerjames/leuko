# TRAINER IMAGE
FROM centos:7

# COMMON DEPENDENCIES
RUN yum install -y gcc vim sudo wget curl git

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
#COPY cudnn-7.0-linux-x64-v3.0-prod.tgz /tmp

RUN cd /tmp && \
    curl -fL http://developer.download.nvidia.com/compute/redist/cudnn/v3/cudnn-7.0-linux-x64-v3.0-prod.tgz > cudnn-7.0-linux-x64-v3.0-prod.tgz && \
    tar -zxvf cudnn-7.0-linux-x64-v3.0-prod.tgz && \
    cp cuda/include/* /usr/local/cuda-7.5/include/ && \
    cp cuda/lib64/* /usr/local/cuda-7.5/lib64/ && \
    rm cudnn-7.0-linux-x64-v3.0-prod.tgz

# CUDA ENVIRONMENT
RUN touch /etc/profile.d/cuda.sh
RUN echo -e "export PATH=/usr/local/cuda-7.5/bin:$PATH" > /etc/profile.d/cuda.sh
RUN echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH" >> /etc/profile.d/cuda.sh

# WORKING DIRECTORY
WORKDIR /root

# EDIT SUDO FILE: DO NOT WANT TO REQUIRE TTY FOR SUDO
RUN sed -i -e "s/Defaults    requiretty.*/ #Defaults    requiretty/g" /etc/sudoers

# TORCH
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch; bash install-deps; ./install.sh

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
RUN cd ~ && \
    git clone https://github.com/soumith/cudnn.torch.git && \
    cd cudnn.torch && \
    git checkout R3 && \
    luarocks make

# DEEP ENVIRONMENT
ENV DEEP_DATA_PATH /root/shared/data
ENV DEEP_SAVE_PATH /root/shared/save
ENV DEEP_UNIT_PATH /root/shared/unit
ENV DEEP_LOG_PATH /root/shared/log

# POSTGRES
RUN yum install -y postgresql postgresql-libs postgresql-devel
RUN echo "hypero-db:5432:*:hypero:hypero" >> ~/.pgpass && \
    chmod og-rwx ~/.pgpass
RUN luarocks install luasql-postgres PGSQL_INCDIR=/usr/include/pgsql/server && \
    luarocks install https://raw.githubusercontent.com/Element-Research/hypero/master/rocks/hypero-scm-1.rockspec
ENV HYPER_PG_CONN "dbname=hypero user=hypero host=hypero-db"

# EXAMPLE FILES
RUN cd ~ && \ 
    mkdir examples && \
    cd examples && \ 
    wget https://raw.githubusercontent.com/Element-Research/hypero/master/examples/neuralnetwork.lua && \
    wget https://raw.githubusercontent.com/nicholas-leonard/dp/master/examples/convolutionneuralnetwork.lua

# HYPERO
RUN cd ~ && \
    git clone https://github.com/Element-Research/hypero.git
