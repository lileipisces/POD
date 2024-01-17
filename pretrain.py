import os
import torch
import argparse
import torch.nn as nn
from transformers import T5Tokenizer
from module import Solomon
from utils import ExpDataLoader, SeqDataLoader, TrainBatchify, ExpBatchify, SeqBatchify, TopNBatchify, now_time


parser = argparse.ArgumentParser(description='POD (PrOmpt Distillation)')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--task_num', type=int, default=3,
                    help='task number')
parser.add_argument('--prompt_num', type=int, default=3,
                    help='prompts per task')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pod/',
                    help='directory to save the final model')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--exp_len', type=int, default=20,
                    help='the maximum length of an explanation')
parser.add_argument('--negative_num', type=int, default=99,
                    help='number of negative items for top-n recommendation')
args = parser.parse_args()

if args.model_version == 1:
    model_version = 't5-base'
elif args.model_version == 2:
    model_version = 't5-large'
elif args.model_version == 3:
    model_version = 't5-3b'
elif args.model_version == 4:
    model_version = 't5-11b'
else:
    model_version = 't5-small'

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
tokenizer = T5Tokenizer.from_pretrained(model_version)
exp_corpus = ExpDataLoader(args.data_dir)
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
all_iterator = TrainBatchify(exp_corpus.train, seq_corpus.user2items_positive, args.negative_num, nitem, tokenizer, args.exp_len, args.batch_size)
exp_iterator = ExpBatchify(exp_corpus.valid, tokenizer, args.exp_len, args.batch_size)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)
topn_iterator = TopNBatchify(seq_corpus.user2items_positive, seq_corpus.user2items_negative, args.negative_num, nitem, tokenizer, args.batch_size)

###############################################################################
# Build the model
###############################################################################

model = Solomon.from_pretrained(model_version)
model.init_prompt(args.task_num, args.prompt_num, device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################


def train():
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        task, source, source_mask, whole_word, target = all_iterator.next_batch()
        task = task.to(device)  # (batch_size,)
        source = source.to(device)  # (batch_size, seq_len)
        source_mask = source_mask.to(device)
        whole_word = whole_word.to(device)
        target = target.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        outputs = model(task, source, whole_word, source_mask, labels=target)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = task.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if all_iterator.batch_index % args.log_interval == 0 or all_iterator.batch_index % all_iterator.batch_num == 0:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'text loss {:4.4f} | {:5d}/{:5d} batches'.format(cur_t_loss, all_iterator.batch_index, all_iterator.batch_num))
            text_loss = 0.
            total_sample = 0
        if all_iterator.batch_index % all_iterator.batch_num == 0:
            break


def evaluate(iterator):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, target = iterator.next_batch_valid()
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            target = target.to(device)
            outputs = model(task, source, whole_word, source_mask, labels=target)
            loss = outputs.loss

            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if iterator.step == iterator.total_step:
                break
    return text_loss / total_sample


with open(model_path, 'wb') as f:
    torch.save(model, f)

print(now_time() + 'Start training')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train()
    print(now_time() + 'validation')
    exp_loss = evaluate(exp_iterator)
    print(now_time() + 'explanation loss {:4.4f}'.format(exp_loss))
    seq_loss = evaluate(seq_iterator)
    print(now_time() + 'sequential loss {:4.4f}'.format(seq_loss))
    topn_loss = evaluate(topn_iterator)
    print(now_time() + 'top-N loss {:4.4f}'.format(topn_loss))
    val_loss = (topn_loss + seq_loss + exp_loss) / 3
    print(now_time() + 'total loss {:4.4f}'.format(val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
