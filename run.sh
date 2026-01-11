docker create --runtime=nvidia --gpus all --net=host --shm-size="32g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2 sleep infinity
docker start verl
docker exec -it verl bash