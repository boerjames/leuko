#! /bin/bash
docker run -it -v /home/grad/boer/shared:/root/shared --link hypero-db:hypero-db --link facetag-db:facetag-db --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 --device /dev/nvidia1 --name learner learner /bin/bash
