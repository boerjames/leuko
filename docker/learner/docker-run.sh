#! /bin/bash
NAME=$1

docker run -it -v /home/grad/boer/shared:/root/shared --link hypero-db:hypero-db --link facetag-db:facetag-db --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 --device /dev/nvidia1 --name $NAME learner /bin/bash
