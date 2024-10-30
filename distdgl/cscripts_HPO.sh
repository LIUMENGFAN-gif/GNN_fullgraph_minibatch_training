NUM_MACHINES=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=1234
WORKER_PER_MACHINE=1
RANK=1
DATASET=ogbn-papers100M
export GLOO_SOCKET_IFNAME=enp225s0
export DGL_DIST_MODE=distributed
export DGL_ROLE=client
export DGL_NUM_SAMPLER=0 
export OMP_NUM_THREADS=10
export DGL_NUM_CLIENT=$((WORKER_PER_MACHINE * NUM_MACHINES))
export DGL_CONF_PATH="/$DATASET/${NUM_MACHINES}part_data/$DATASET.json"
export DGL_IP_CONFIG=ip_config.txt
export DGL_GRAPH_FORMAT=csc
export DGL_NUM_SERVER=1
torchrun --nproc_per_node=$WORKER_PER_MACHINE --nnodes=$NUM_MACHINES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main_HPO_cross_machines.py \
--isClient True \
--num_parts $NUM_MACHINES \
--rank $RANK \
--not_time_record True \
--start_dev 0 \
--name $DATASET \
--batch_size 50000 \
--lr 0.001 \
--drop_rate 0 \
--weight_decay 0 \
--fanouts 5,10,15 \
--num_epochs 10 \
--patience 10 \
--num_gpus $WORKER_PER_MACHINE \
--count 48 \
--start_dev 0
