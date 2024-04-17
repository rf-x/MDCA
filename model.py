import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, GenerationConfig, T5ForConditionalGeneration


class MDCABartModel(nn.Module):
    def __init__(self, args) -> None:
        super(MDCABartModel, self).__init__()
        self.args = args

        self.bart = BartForConditionalGeneration.from_pretrained(args.pretrained_model_dir)
        self.bart.resize_token_embeddings(len(args.tokenizer))
        self.text_embeddings = self.bart.get_input_embeddings()
        
        self.img_fc = nn.Linear(args.img_hidden_size, args.hidden_size)
        self.classify_fc = nn.Linear(args.hidden_size, args.num_classes)

        self.ea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, 'ea_generation_config.json')
        self.iea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, 'iea_generation_config.json')

    def forward(self, a_input_ids, a_attention_mask, cls_indexer, ea_input_ids, ea_attention_mask, ea_decoder_output_labels, iea_input_ids, iea_attention_mask, iea_decoder_output_labels, image_feature, is_eval=False):
        img_feat = self.img_fc(image_feature)

        a_encoder_inputs_embeds = self.text_embeddings(a_input_ids)   # (B, L, H)
        a_encoder_inputs_embeds = torch.cat([a_encoder_inputs_embeds[:, :3, :], img_feat, a_encoder_inputs_embeds[:, 3:, :]], dim=1)  # <s>qa: <img>img_feat

        ea_encoder_inputs_embeds = self.text_embeddings(ea_input_ids)   # (B, L, H)
        ea_encoder_inputs_embeds = torch.cat([ea_encoder_inputs_embeds[:, :3, :], img_feat, ea_encoder_inputs_embeds[:, 3:, :]], dim=1)  # <s>qea: <img>img_feat

        iea_encoder_inputs_embeds = self.text_embeddings(iea_input_ids)   # (B, L, H)
        iea_encoder_inputs_embeds = torch.cat([iea_encoder_inputs_embeds[:, :3, :], img_feat, iea_encoder_inputs_embeds[:, 3:, :]], dim=1)  # <s>qiea: <img>img_feat

        a_bart_output = self.bart(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask,decoder_inputs_embeds=a_encoder_inputs_embeds[:,:-1,:], decoder_attention_mask=a_attention_mask[:,:-1], output_hidden_states=True)
        last_hidden_state = a_bart_output.decoder_hidden_states[-1]  # (B, L, H)
        cls_hidden_state = torch.stack([torch.index_select(f, 0, w_i) for f, w_i in zip(last_hidden_state, cls_indexer)]).squeeze(1)
        a_logits = self.classify_fc(cls_hidden_state)

        if not is_eval:
            ea_bart_output = self.bart(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, labels=ea_decoder_output_labels)
            ea_loss = ea_bart_output.loss

            iea_bart_output = self.bart(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, labels=iea_decoder_output_labels)
            iea_loss = iea_bart_output.loss

            return a_logits, ea_loss, iea_loss

        else:
            ea_sequence_ids = self.bart.generate(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, generation_config=self.ea_generation_config)
            iea_sequence_ids = self.bart.generate(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, generation_config=self.iea_generation_config)
            ea_sequence = self.args.tokenizer.batch_decode(ea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            iea_sequence = self.args.tokenizer.batch_decode(iea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return a_logits, ea_sequence, iea_sequence
    
        
class MDCAFlanT5Model(nn.Module):
    def __init__(self, args) -> None:
        super(MDCAFlanT5Model, self).__init__()
        self.args = args

        self.t5 = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_dir)
        self.t5.resize_token_embeddings(len(args.tokenizer))
        self.text_embeddings = self.t5.get_input_embeddings()

        self.img_fc = nn.Linear(args.img_hidden_size, args.hidden_size)

        self.a_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, 'a_generation_config.json')
        self.ea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, 'ea_generation_config.json')
        self.iea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, 'iea_generation_config.json')

    def forward(self, a_input_ids, a_attention_mask, a_decoder_output_labels, ea_input_ids, ea_attention_mask, ea_decoder_output_labels, iea_input_ids,
                iea_attention_mask, iea_decoder_output_labels, image_feature, is_eval=False):
        img_feat = self.img_fc(image_feature)

        a_encoder_inputs_embeds = self.text_embeddings(a_input_ids)  # (B, L, H)
        a_encoder_inputs_embeds = torch.cat([a_encoder_inputs_embeds[:, :2, :], img_feat, a_encoder_inputs_embeds[:, 2:, :]], dim=1)  # qa: <img>img_feat

        ea_encoder_inputs_embeds = self.text_embeddings(ea_input_ids)  # (B, L, H)
        ea_encoder_inputs_embeds = torch.cat([ea_encoder_inputs_embeds[:, :2, :], img_feat, ea_encoder_inputs_embeds[:, 2:, :]], dim=1)  # qea: <img>img_feat


        iea_encoder_inputs_embeds = self.text_embeddings(iea_input_ids)  # (B, L, H)
        iea_encoder_inputs_embeds = torch.cat([iea_encoder_inputs_embeds[:, :2, :], img_feat, iea_encoder_inputs_embeds[:, 2:, :]], dim=1)  # qiea: <img>img_feat

        if not is_eval: 
            a_t5_output = self.t5(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask, labels=a_decoder_output_labels)
            a_loss = a_t5_output.loss

            ea_t5_output = self.t5(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, labels=ea_decoder_output_labels)
            ea_loss = ea_t5_output.loss

            iea_t5_output = self.t5(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, labels=iea_decoder_output_labels)
            iea_loss = iea_t5_output.loss

            return a_loss, ea_loss, iea_loss

        else:
            a_sequence_ids = self.t5.generate(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask, generation_config=self.a_generation_config)
            ea_sequence_ids = self.t5.generate(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, generation_config=self.ea_generation_config)
            iea_sequence_ids = self.t5.generate(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, generation_config=self.iea_generation_config)
            a_sequence = self.args.tokenizer.batch_decode(a_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ea_sequence = self.args.tokenizer.batch_decode(ea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            iea_sequence = self.args.tokenizer.batch_decode(iea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return a_sequence, ea_sequence, iea_sequence


