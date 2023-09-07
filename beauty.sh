CUDA_VISIBLE_DEVICES=1 python -u pretrain.py \
--data_dir ./data/beauty/ \
--cuda \
--batch_size 64 \
--model_version 0 \
--checkpoint ./checkpoint/beauty/ \
--lr 0.0005

CUDA_VISIBLE_DEVICES=1 python -u seq.py \
--data_dir ./data/beauty/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/beauty/

CUDA_VISIBLE_DEVICES=1 python -u topn.py \
--data_dir ./data/beauty/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/beauty/

CUDA_VISIBLE_DEVICES=1 python -u exp.py \
--data_dir ./data/beauty/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/beauty/