CUDA_VISIBLE_DEVICES=3 python -u pretrain.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 64 \
--model_version 0 \
--checkpoint ./checkpoint/toys/ \
--lr 0.0005

CUDA_VISIBLE_DEVICES=3 python -u seq.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/toys/

CUDA_VISIBLE_DEVICES=3 python -u topn.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/toys/

CUDA_VISIBLE_DEVICES=3 python -u exp.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/toys/