# Script to freeze models from github.com/huggingface/pytorch-pretrained-BERT
# to create ONNX files.
#
# NOTE: This script doesn't export properly with current PyTorch

import torch
from pytorch_pretrained_bert import TransfoXLLMHeadModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=1,type=int,help='Batch size for inference')

    parser.add_argument('--model_name',default='transfo-xl-wt103',type=str,
                        help='Pre-trained model name')
    parser.add_argument('--max_seq_length',default=128,type=int,
                        help='Maximum total input sequence length after tokenization')

    args = parser.parse_args()

    input_ids = torch.zeros([args.batch_size,args.max_seq_length],dtype=torch.long)

    model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
    torch.onnx.export(model,input_ids,'transfoxll_'+'batch'+str(args.batch_size)+'.onnx')

if __name__ == "__main__":
    main()
