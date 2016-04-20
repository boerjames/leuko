#! /bin/bash
docker run -it -d --name hypero-db hypero-db
docker exec -it -d hypero-db sudo -u postgres service postgresql restart
