import os
import torch
import argparse
from transformers import T5Tokenizer
from utils import rouge_score, bleu_score, ExpDataLoader, ExpBatchify, now_time, ids2tokens


parser = argparse.ArgumentParser(description='POD (PrOmpt Distillation)')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./pod/',
                    help='directory to load the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--num_beams', type=int, default=21,
                    help='number of beams')
parser.add_argument('--num_beam_groups', type=int, default=3,
                    help='number of beam groups')
parser.add_argument('--min_len', type=int, default=10,
                    help='the minimum length of an explanation')
parser.add_argument('--exp_len', type=int, default=20,
                    help='the maximum length of an explanation')
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
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
tokenizer = T5Tokenizer.from_pretrained(model_version)
exp_corpus = ExpDataLoader(args.data_dir)
exp_iterator = ExpBatchify(exp_corpus.test, tokenizer, args.exp_len, args.batch_size)

###############################################################################
# Test the model
###############################################################################


def generate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, _ = exp_iterator.next_batch_test()
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)

            beam_outputs = model.my_beam_search(task, source, whole_word, source_mask,
                                             min_length=args.min_len,
                                             num_beams=args.num_beams,
                                             num_beam_groups=args.num_beam_groups,
                                             num_return_sequences=1
                                             )

            idss_predict.extend(beam_outputs.tolist())

            if exp_iterator.step == exp_iterator.total_step:
                break
    return idss_predict


# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


print(now_time() + 'Generating text')
idss_predicted = generate()
print(now_time() + 'Evaluation')
tokens_test = [ids2tokens(ids, tokenizer) for ids in exp_iterator.target_seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
