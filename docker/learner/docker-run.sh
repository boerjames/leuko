docker run -it -v /home/grad/boer/deep:/root/deep --link hypero-db:hypero-db --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 --device /dev/nvidia1 --name $2 $1 /bin/bash