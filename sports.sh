CUDA_VISIBLE_DEVICES=2 python -u pretrain.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 64 \
--model_version 0 \
--checkpoint ./checkpoint/sports/ \
--lr 0.001

CUDA_VISIBLE_DEVICES=2 python -u seq.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/sports/

CUDA_VISIBLE_DEVICES=2 python -u topn.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/sports/

CUDA_VISIBLE_DEVICES=2 python -u exp.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/sports/