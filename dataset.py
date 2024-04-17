import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import read_json

class Twitter_Dataset_Bart(Dataset):
    def __init__(self, args, split):
        # train.json captions.json images_feature
        self.args = args
        self.data_path = os.path.join(args.data_dir, args.dataset_name)
        self.captions_path = os.path.join(self.data_path, 'captions.json')
        self.images_feature_path = os.path.join(self.data_path, 'images_feature')
        
        self.caption_data = self.get_captions(self.captions_path)
    
        if split == 'train':
            self.data_set = json.load(
                open(self.data_path + '/train_cause.json', 'r'))
        elif split == 'dev':
            self.data_set = json.load(
                open(self.data_path + '/dev_cause.json', 'r'))
        elif split == 'test':
            self.data_set = json.load(
                open(self.data_path + '/test_cause.json', 'r'))
        else:
            raise RuntimeError("split type is not exist!!!")

    def __len__(self):
        return len(self.data_set)

    def get_captions(self, data_path):
        data = read_json(data_path)
        return data
    
    def get_image_feature(self, image_id):
        image_feature = np.load(os.path.join(self.images_feature_path, image_id[:-4] + '.npz'))['embedding']
        return image_feature
    
    def get_input_sentence(self, sentence, aspect, caption):
        a_input_sentence = "qa: <image></image> caption: {} ".format(caption) + "sentence: {} aspect: {}".format(sentence, aspect)
        ea_input_sentence = "qea: <image></image> caption: {} ".format(caption) + "sentence: {} aspect: {}.".format(sentence, aspect)
        iea_input_sentence = "qiea: <image></image> caption: {} ".format(caption) + "sentence: {} aspect: {}.".format(sentence, aspect)
        return a_input_sentence, ea_input_sentence, iea_input_sentence
    
    def get_output_sentence(self, label, explanation, i_explanation):
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(label)]
        ea_output_sentence = '<explain>{}</explain><emotion>{}</emotion>'.format(explanation, sentiment)
        iea_output_sentence = '<i_explain>{}</i_explain><emotion>{}</emotion>'.format(i_explanation, sentiment)
        return ea_output_sentence, iea_output_sentence

    def __getitem__(self, index):
        data = self.data_set[index]
        image_id = data['image']
        sentiment_label = data['label'] 
        captions = self.caption_data[image_id] 
        image_feature= self.get_image_feature(image_id)  # np (1,196,768)
        
        a_input_sentence, ea_input_sentence, iea_input_sentence = self.get_input_sentence(data['sentence'], data['aspect'], captions[self.args.cap_index])
        ea_output_sentence, iea_output_sentence = self.get_output_sentence(sentiment_label, data['response'], data['image_response'])
        
        # input
        a_input_tokens = self.args.tokenizer.tokenize(a_input_sentence)
        a_input_ids = self.args.tokenizer.convert_tokens_to_ids(a_input_tokens)
        ea_input_tokens = self.args.tokenizer.tokenize(ea_input_sentence)
        ea_input_ids = self.args.tokenizer.convert_tokens_to_ids(ea_input_tokens)
        iea_input_tokens = self.args.tokenizer.tokenize(iea_input_sentence)
        iea_input_ids = self.args.tokenizer.convert_tokens_to_ids(iea_input_tokens)

        # output
        ea_output_tokens = self.args.tokenizer.tokenize(ea_output_sentence)
        ea_output_ids = [self.args.tokenizer.bos_token_id] + self.args.tokenizer.convert_tokens_to_ids(ea_output_tokens) 
        ea_output_labels = ea_output_ids[1:] + [self.args.tokenizer.eos_token_id]
        iea_output_tokens = self.args.tokenizer.tokenize(iea_output_sentence)
        iea_output_ids = [self.args.tokenizer.bos_token_id] + self.args.tokenizer.convert_tokens_to_ids(iea_output_tokens) 
        iea_output_labels = iea_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        a_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(a_input_ids) #  <s> X </s>
        ea_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(ea_input_ids) #  <s> X </s>
        iea_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(iea_input_ids) #  <s> X </s>

        # attention mask, 196 is the length of image features
        cls_indexer = [len(a_input_ids) + 196 - 1 - 1]
        a_attention_mask = [1] * (len(a_input_ids) + 196)
        ea_attention_mask = [1] * (len(ea_input_ids) + 196)
        iea_attention_mask = [1] * (len(iea_input_ids) + 196)
    
        return (torch.tensor(a_input_ids), torch.tensor(a_attention_mask), torch.tensor(cls_indexer),
                torch.tensor(ea_input_ids), torch.tensor(ea_attention_mask), torch.tensor(ea_output_labels), 
                torch.tensor(iea_input_ids), torch.tensor(iea_attention_mask), torch.tensor(iea_output_labels), 
                torch.from_numpy(image_feature), torch.tensor(sentiment_label))
        


class Twitter_Dataset_FlanT5(Dataset):
    def __init__(self, args, split):

        self.args = args
        self.data_path = os.path.join(args.data_dir, args.dataset_name)
        self.captions_path = os.path.join(self.data_path, 'captions.json')
        self.images_feature_path = os.path.join(self.data_path, 'images_feature')

        self.caption_data = self.get_captions(self.captions_path)

        if split == 'train':
            self.data_set = json.load(
                open(self.data_path + '/train_causeg.json', 'r'))
        elif split == 'dev':
            self.data_set = json.load(
                open(self.data_path + '/dev_cause.json', 'r'))
        elif split == 'test':
            self.data_set = json.load(
                open(self.data_path + '/test_cause.json', 'r'))
        else:
            raise RuntimeError("split type is not exist!!!")

    def __len__(self):
        return len(self.data_set)

    def get_captions(self, data_path):
        data = read_json(data_path)
        return data

    def get_image_feature(self, image_id):
        image_feature = np.load(os.path.join(self.images_feature_path, image_id[:-4] + '.npz'))['embedding']
        return image_feature

    def get_input_sentence(self, sentence, aspect, caption):
        a_input_sentence = "qa: <image></image> caption: {} ".format(caption) + "sentence: {} aspect: {}".format(sentence, aspect)
        ea_input_sentence = "qea: <image></image> caption: {} ".format(caption) + "sentence: {} aspect: {}.".format(sentence, aspect)
        iea_input_sentence = "qiea: <image></image> caption: {} ".format(caption) + "sentence: {} aspect: {}.".format(sentence, aspect)
        return a_input_sentence, ea_input_sentence, iea_input_sentence

    def get_output_sentence(self, label, explanation, i_explanation):
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(label)]
        a_output_sentence = '<emotion>{}</emotion>'.format(sentiment)
        ea_output_sentence = '<explain>{}</explain><emotion>{}</emotion>'.format(explanation, sentiment)
        iea_output_sentence = '<i_explain>{}</i_explain><emotion>{}</emotion>'.format(i_explanation, sentiment)
        return a_output_sentence, ea_output_sentence, iea_output_sentence


    def __getitem__(self, index):
        data = self.data_set[index]
        image_id = data['image']
        sentiment_label = data['label']
        captions = self.caption_data[image_id]  
        image_feature= self.get_image_feature(image_id)  # np (1,196,768)
        a_input_sentence, ea_input_sentence, iea_input_sentence = self.get_input_sentence(data['sentence'], data['aspect'], captions[self.args.cap_index])
        a_output_sentence, ea_output_sentence, iea_output_sentence = self.get_output_sentence(sentiment_label, data['response'], data['image_response'])
        
        a_input_tokens = self.args.tokenizer.tokenize(a_input_sentence)
        a_input_ids = self.args.tokenizer.convert_tokens_to_ids(a_input_tokens)
        ea_input_tokens = self.args.tokenizer.tokenize(ea_input_sentence)
        ea_input_ids = self.args.tokenizer.convert_tokens_to_ids(ea_input_tokens)
        iea_input_tokens = self.args.tokenizer.tokenize(iea_input_sentence)
        iea_input_ids = self.args.tokenizer.convert_tokens_to_ids(iea_input_tokens)

        a_output_tokens = self.args.tokenizer.tokenize(a_output_sentence)
        a_output_ids = [self.args.tokenizer.pad_token_id] + self.args.tokenizer.convert_tokens_to_ids(a_output_tokens)
        a_output_labels = a_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        ea_output_tokens = self.args.tokenizer.tokenize(ea_output_sentence)
        ea_output_ids = [self.args.tokenizer.pad_token_id] + self.args.tokenizer.convert_tokens_to_ids(ea_output_tokens)
        ea_output_labels = ea_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        iea_output_tokens = self.args.tokenizer.tokenize(iea_output_sentence)
        iea_output_ids = [self.args.tokenizer.pad_token_id] + self.args.tokenizer.convert_tokens_to_ids(iea_output_tokens)
        iea_output_labels = iea_output_ids[1:] + [self.args.tokenizer.eos_token_id]

        a_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(a_input_ids)  # X </s>
        ea_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(ea_input_ids)  # X </s>
        iea_input_ids = self.args.tokenizer.build_inputs_with_special_tokens(iea_input_ids)  # X </s>

        
         # attention mask, 196 is the length of image features
        a_attention_mask = [1] * (len(a_input_ids) + 196)
        ea_attention_mask = [1] * (len(ea_input_ids) + 196)
        iea_attention_mask = [1] * (len(iea_input_ids) + 196)
        
        return (torch.tensor(a_input_ids), torch.tensor(a_attention_mask), torch.tensor(a_output_labels),
                torch.tensor(ea_input_ids), torch.tensor(ea_attention_mask), torch.tensor(ea_output_labels),
                torch.tensor(iea_input_ids), torch.tensor(iea_attention_mask), torch.tensor(iea_output_labels),
                torch.from_numpy(image_feature), torch.tensor(sentiment_label))



def collate_fn_bart(batch):
    '''
    Pad sentence a batch.
    Turn all into tensors.
    '''
    a_input_ids, a_attention_mask, cls_indexer, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels = zip(*batch)

    a_input_ids = pad_sequence(a_input_ids, batch_first=True, padding_value=1)
    cls_indexer = torch.tensor(cls_indexer)
    ea_input_ids = pad_sequence(ea_input_ids, batch_first=True, padding_value=1)
    ea_output_labels = pad_sequence(ea_output_labels, batch_first=True, padding_value=-100)
    iea_input_ids = pad_sequence(iea_input_ids, batch_first=True, padding_value=1)
    iea_output_labels = pad_sequence(iea_output_labels, batch_first=True, padding_value=-100)
    
    image_feature = pad_sequence(image_feature, batch_first=True, padding_value=0)
     
    a_attention_mask = pad_sequence(a_attention_mask, batch_first=True, padding_value=0)
    ea_attention_mask= pad_sequence(ea_attention_mask, batch_first=True, padding_value=0)
    iea_attention_mask= pad_sequence(iea_attention_mask, batch_first=True, padding_value=0)
    
    sentiment_labels = torch.tensor(sentiment_labels)

    return a_input_ids, a_attention_mask, cls_indexer, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels


def collate_fn_flant5(batch):
    '''
    Pad sentence a batch.
    Turn all into tensors.
    '''
    a_input_ids, a_attention_mask, a_output_labels, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels = zip(*batch)

    a_input_ids = pad_sequence(a_input_ids, batch_first=True, padding_value=0)
    a_output_labels = pad_sequence(a_output_labels, batch_first=True, padding_value=-100)
    ea_input_ids = pad_sequence(ea_input_ids, batch_first=True, padding_value=0)
    ea_output_labels = pad_sequence(ea_output_labels, batch_first=True, padding_value=-100)
    iea_input_ids = pad_sequence(iea_input_ids, batch_first=True, padding_value=0)
    iea_output_labels = pad_sequence(iea_output_labels, batch_first=True, padding_value=-100)

    image_feature = pad_sequence(image_feature, batch_first=True, padding_value=0)

    a_attention_mask = pad_sequence(a_attention_mask, batch_first=True, padding_value=0)
    ea_attention_mask = pad_sequence(ea_attention_mask, batch_first=True, padding_value=0)
    iea_attention_mask = pad_sequence(iea_attention_mask, batch_first=True, padding_value=0)

    sentiment_labels = torch.tensor(sentiment_labels)

    return a_input_ids, a_attention_mask, a_output_labels, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature, sentiment_labels
     





