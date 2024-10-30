NUM_MACHINES=2
WORKER_PER_MACHINE=4
RANK=1
DATASET=ogbn-papers100M
export GLOO_SOCKET_IFNAME=enp225s0
export DGL_DIST_MODE=distributed
export DGL_ROLE=server
export DGL_NUM_SAMPLER=0 
export OMP_NUM_THREADS=1
export DGL_NUM_CLIENT=$((WORKER_PER_MACHINE * NUM_MACHINES))
export DGL_CONF_PATH="$DATASET/bidirected_${NUM_MACHINES}part_data/$DATASET.json"
export DGL_IP_CONFIG=ip_config.txt
export DGL_GRAPH_FORMAT=csc
export DGL_NUM_SERVER=1
export DGL_SERVER_ID=$RANK
python main_cross_machines.py \
--isClient True \
--num_parts $((WORKER_PER_MACHINE * NUM_MACHINES)) \
--rank $RANK \
--not_time_record True \
--start_dev 0 \
--name $DATASET \
--batch_size 1280 \
--lr 0.0001 \
--drop_rate 0.5 \
--weight_decay 0.001 \
--fanouts 25,15 \
--num_epochs 100000 \
--patience 10 \
--num_gpus $WORKER_PER_MACHINE \
--hidden_feats 2048 \
--num_layers 2