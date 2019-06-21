# Script to freeze models from github.com/huggingface/pytorch-pretrained-BERT
# to create ONNX files.
#
# NOTE: Only some of the models in repository are fully trained.  Hence, the
#       ONNX files exported for those models are marked as "_untrained.onnx"
#       trained versions are possible if export commands such as those below
#       are inserted at appropriate spots in the training scripts.

import torch
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining, BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=1,type=int,help='Batch size for inference')

    parser.add_argument('--bert_model',default='bert-base-cased',type=str,
                        help='Bert pre-trained model selected, e.g. bert-base-uncased, bert-large-uncased, bert-base-multilingual-case, bert-base-chinese')
    parser.add_argument('--max_seq_length',default=128,type=int,
                        help='Maximum total input sequence length after tokenization')

    args = parser.parse_args()

    input_ids = torch.zeros([args.batch_size,args.max_seq_length],dtype=torch.long)
    token_type_ids = torch.zeros([args.batch_size,args.max_seq_length],dtype=torch.long)    

    # Export various BERT models
    # Note: For argument definitions used here see modeling.py from pytorch-pretrained-bert
    #       repository
    # 
    # Fully trained models
    model = BertModel.from_pretrained(args.bert_model)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.onnx')

    model = BertForMaskedLM.from_pretrained(args.bert_model)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_maskedlm_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.onnx')

    model = BertForNextSentencePrediction.from_pretrained(args.bert_model)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_nextsentence_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.onnx')

    model = BertForPreTraining.from_pretrained(args.bert_model)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_pretraining_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.onnx')

    # Partially trained models
    model = BertForSequenceClassification.from_pretrained(args.bert_model,2)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_classify_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.untrained.onnx')

    model = BertForTokenClassification.from_pretrained(args.bert_model,2)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_tokenclassify_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.untrained.onnx')

    model = BertForQuestionAnswering.from_pretrained(args.bert_model,2)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_question_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.untrained.onnx')    

    choices=2
    input_ids = torch.zeros([args.batch_size,choices,args.max_seq_length],dtype=torch.long)
    token_type_ids = torch.zeros([args.batch_size,choices,args.max_seq_length],dtype=torch.long)        
    model = BertForMultipleChoice.from_pretrained(args.bert_model,choices)
    torch.onnx.export(model,(input_ids,token_type_ids),'bert_multiplechoice_'+'batch'+str(args.batch_size)+'_'+args.bert_model+'.untrained.onnx')

if __name__ == "__main__":
    main()
