# Script to freeze models from github.com/huggingface/pytorch-pretrained-BERT
# to create ONNX files.
#
# NOTE: This generates warnings when the ONNX file is created, e.g.
#
# TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect.
# We can't record the data flow of Python values, so this value will be treated as a constant
# in the future. This means that the trace might not generalize to other inputs!
# b = self.bias[:, :, : w.size(-2), : w.size(-1)]
#
# Unclear if this is missing parameters or some other issue.


import torch
from pytorch_pretrained_bert import GPT2LMHeadModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=1,type=int,help='Batch size for inference')

    parser.add_argument('--model_name',default='gpt2',type=str,
                        help='Pre-trained model name')
    parser.add_argument('--max_seq_length',default=128,type=int,
                        help='Maximum total input sequence length after tokenization')

    args = parser.parse_args()

    input_ids = torch.zeros([args.batch_size,args.max_seq_length],dtype=torch.long)

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    torch.onnx.export(model,input_ids,'gpt2_'+'batch'+str(args.batch_size)+'.onnx')

if __name__ == "__main__":
    main()
