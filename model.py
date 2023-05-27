"""
Name: MTN
Date: 2022/10/29
Version: 1.0
"""

import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import os
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
import matplotlib.pyplot as plt
from pre_model import RobertaEncoder
import copy
# from Orthographic_pytorch import Ortho_algorithm_unique,Ortho_algorithm_common
from Vit3 import ViT
from torch.nn import TransformerEncoderLayer, Transformer, MultiheadAttention
from sentence_transformers import SentenceTransformer
from data_process import data_process
from transformers import ViTModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, text=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.text = text

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, text=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.text = text


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


# def l1norm(X, dim, eps=1e-8):
#     """L1-normalize columns of X
#     """
#     norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
#     X = torch.div(X, norm)
#     return X
#
#
# def l2norm(X, dim, eps=1e-8):
#     """L2-normalize columns of X
#     """
#     norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
#     X = torch.div(X, norm)
#     return X
#
#
# def func_attention(query, context, opt, smooth, eps=1e-8):
#     """
#     query: (n_context, queryL, d)
#     context: (n_context, sourceL, d)
#     """
#     batch_size_q, queryL = query.size(0), query.size(1)
#     batch_size, sourceL = context.size(0), context.size(1)
#
#     # Get attention
#     # --> (batch, d, queryL)
#     queryT = torch.transpose(query, 1, 2)
#
#     # (batch, sourceL, d)(batch, d, queryL)
#     # --> (batch, sourceL, queryL)
#     attn = torch.bmm(context, queryT)
#     if opt.raw_feature_norm == "softmax":
#         # --> (batch*sourceL, queryL)
#         attn = attn.view(batch_size * sourceL, queryL)
#         attn = nn.Softmax()(attn)
#         # --> (batch, sourceL, queryL)
#         attn = attn.view(batch_size, sourceL, queryL)
#     elif opt.raw_feature_norm == "l2norm":
#         attn = l2norm(attn, 2)
#     elif opt.raw_feature_norm == "clipped_l2norm":
#         attn = nn.LeakyReLU(0.1)(attn)
#         attn = l2norm(attn, 2)
#     elif opt.raw_feature_norm == "clipped":
#         attn = nn.LeakyReLU(0.1)(attn)
#     elif opt.raw_feature_norm == "no_norm":
#         pass
#     else:
#         raise ValueError("unknown first norm type:", opt.raw_feature_norm)
#     # --> (batch, queryL, sourceL)
#     attn = torch.transpose(attn, 1, 2).contiguous()
#     # --> (batch*queryL, sourceL)
#     attn = attn.view(batch_size * queryL, sourceL)
#     attn = nn.Softmax()(attn * smooth)
#     # --> (batch, queryL, sourceL)
#     attn = attn.view(batch_size, queryL, sourceL)
#     # --> (batch, sourceL, queryL)
#     attnT = torch.transpose(attn, 1, 2).contiguous()
#
#     # --> (batch, d, sourceL)
#     contextT = torch.transpose(context, 1, 2)
#     # (batch x d x sourceL)(batch x sourceL x queryL)
#     # --> (batch, d, queryL)
#     weightedContext = torch.bmm(contextT, attnT)
#     # --> (batch, queryL, d)
#     weightedContext = torch.transpose(weightedContext, 1, 2)
#
#     return weightedContext, attnT
#
#
# def cosine_similarity(x1, x2, dim=1, eps=1e-8):
#     """Returns cosine similarity between x1 and x2, computed along dim."""
#     w12 = torch.sum(x1 * x2, dim)
#     w1 = torch.norm(x1, 2, dim)
#     w2 = torch.norm(x2, 2, dim)
#     return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
#
#
# def xattn_score_t2i(images, captions, cap_lens, opt):
#     """
#     Images: (n_image, n_regions, d) matrix of images
#     Captions: (n_caption, max_n_word, d) matrix of captions
#     CapLens: (n_caption) array of caption lengths
#     """
#     similarities = []
#     n_image = images.size(0)
#     n_caption = captions.size(0)
#     for i in range(n_caption):
#         # Get the i-th text description
#         n_word = cap_lens[i]
#         cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
#         # --> (n_image, n_word, d)
#         cap_i_expand = cap_i.repeat(n_image, 1, 1)
#         """
#             word(query): (n_image, n_word, d)
#             image(context): (n_image, n_regions, d)
#             weiContext: (n_image, n_word, d)
#             attn: (n_image, n_region, n_word)
#         """
#         weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
#         cap_i_expand = cap_i_expand.contiguous()
#         weiContext = weiContext.contiguous()
#         # (n_image, n_word)
#         row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
#         if opt.agg_func == 'LogSumExp':
#             row_sim.mul_(opt.lambda_lse).exp_()
#             row_sim = row_sim.sum(dim=1, keepdim=True)
#             row_sim = torch.log(row_sim) / opt.lambda_lse
#         elif opt.agg_func == 'Max':
#             row_sim = row_sim.max(dim=1, keepdim=True)[0]
#         elif opt.agg_func == 'Sum':
#             row_sim = row_sim.sum(dim=1, keepdim=True)
#         elif opt.agg_func == 'Mean':
#             row_sim = row_sim.mean(dim=1, keepdim=True)
#         else:
#             raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
#         similarities.append(row_sim)
#
#     # (n_image, n_caption)
#     similarities = torch.cat(similarities, 1)
#
#     return similarities
#
#
# def xattn_score_i2t(images, captions, cap_lens, opt):
#     """
#     Images: (batch_size, n_regions, d) matrix of images
#     Captions: (batch_size, max_n_words, d) matrix of captions
#     CapLens: (batch_size) array of caption lengths
#     """
#     similarities = []
#     n_image = images.size(0)
#     n_caption = captions.size(0)
#     n_region = images.size(1)
#     for i in range(n_caption):
#         # Get the i-th text description
#         n_word = cap_lens[i]
#         cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
#         # (n_image, n_word, d)
#         cap_i_expand = cap_i.repeat(n_image, 1, 1)
#         """
#             word(query): (n_image, n_word, d)
#             image(context): (n_image, n_region, d)
#             weiContext: (n_image, n_region, d)
#             attn: (n_image, n_word, n_region)
#         """
#         weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
#         # (n_image, n_region)
#         row_sim = cosine_similarity(images, weiContext, dim=2)
#         if opt.agg_func == 'LogSumExp':
#             row_sim.mul_(opt.lambda_lse).exp_()
#             row_sim = row_sim.sum(dim=1, keepdim=True)
#             row_sim = torch.log(row_sim) / opt.lambda_lse
#         elif opt.agg_func == 'Max':
#             row_sim = row_sim.max(dim=1, keepdim=True)[0]
#         elif opt.agg_func == 'Sum':
#             row_sim = row_sim.sum(dim=1, keepdim=True)
#         elif opt.agg_func == 'Mean':
#             row_sim = row_sim.mean(dim=1, keepdim=True)
#         else:
#             raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
#         similarities.append(row_sim)
#
#     # (n_image, n_caption)
#     similarities = torch.cat(similarities, 1)
#     return similarities
#
#
# class ContrastiveLoss(nn.Module):
#     """
#     Compute contrastive loss
#     """
#
#     def __init__(self, opt, margin=0, max_violation=False):
#         super(ContrastiveLoss, self).__init__()
#         self.opt = opt
#         self.margin = margin
#         self.max_violation = max_violation
#
#     def forward(self, im, s, s_l):
#         # compute image-sentence score matrix
#         if self.opt.cross_attn == 't2i':
#             scores = xattn_score_t2i(im, s, s_l, self.opt)
#         elif self.opt.cross_attn == 'i2t':
#             scores = xattn_score_i2t(im, s, s_l, self.opt)
#         else:
#             raise ValueError("unknown first norm type:", "clipped_l2norm")
#         diagonal = scores.diag().view(im.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)
#
#         # compare every diagonal score to scores in its column
#         # caption retrieval
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         # compare every diagonal score to scores in its row
#         # image retrieval
#         cost_im = (self.margin + scores - d2).clamp(min=0)
#
#         # clear diagonals
#         mask = torch.eye(scores.size(0)) > .5
#         I = Variable(mask)
#         if torch.cuda.is_available():
#             I = I.cuda()
#         cost_s = cost_s.masked_fill_(I, 0)
#         cost_im = cost_im.masked_fill_(I, 0)
#
#         # keep the maximum violating negative for each query
#         if self.max_violation:
#             cost_s = cost_s.max(1)[0]
#             cost_im = cost_im.max(0)[0]
#         return cost_s.sum() + cost_im.sum()


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = ''

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = False  # True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        # input = input.texts
        output = self.model(input, attention_mask=attention_mask)
        return output


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_model = TextModel(opt)
        self.vit = ViTModel.from_pretrained('facebook/deit-base-patch16-224')
        self.text_config = copy.deepcopy(self.text_model.get_config())
        self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.image_encoder = RobertaEncoder(self.image_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            # nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            nn.Linear(2048, opt.tran_dim),
            ActivateFun(opt)
        )  # 2048
        self.image_cls_change = nn.Sequential(
            # nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            nn.Linear(2048, opt.tran_dim),
            ActivateFun(opt)
        )  # 2048
        # self.encoder_layer = TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer = Transformer(nhead=1, num_encoder_layers=1, num_decoder_layers=1, d_model=768, dim_feedforward=128)  # 4,1,1,768,128;

        self.multiheadattention = MultiheadAttention(768, 16, dropout=0.1)
        # self.multiheadattention = layers.MultiHeadAttention(head, dim, dim, dim, dropout=dropout)
        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)
        # self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_transformer = SentenceTransformer('microsoft/mpnet-base')


        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)  # 此处需要注意
        )


    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask, text):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        # text_encoder = self.text_model(text_inputs.texts, attention_mask=bert_attention_mask.bert_attention_mask)
        # sentence = self.sentence_transformer.encode(text, convert_to_tensor=True)
        # sentence = sentence.unsqueeze(dim=0).cuda()
        # text_cls = text_encoder.pooler_output

        # print('text: ', text)
        # exit()

        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)
        # vit模型
        image_feature = self.vit(image_inputs)
        image_init = image_feature.last_hidden_state
        # image_encoder, image_cls = self.v(image_inputs)
        for param in self.vit.parameters():
            param.requires_grad = False

        # image_init = image_init.permute(1, 0, 2).contiguous()
        # text_init = text_init.permute(1, 0, 2).contiguous()
        # text_image_cat = self.transformer(image_init, text_init)
        # # text_image_cat = self.transformer(sentence, text_image_cat)
        # text_image_transformer = text_image_cat.permute(1, 2, 0).contiguous()
        # concat
        text_image_transformer = torch.cat((text_init, image_init), dim=1)
        text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()


        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()

            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:text_image_output.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention(text_image_output)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

            text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(2)
            text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
        else:
            raise Exception('fuse_type设定错误')
        cap_length = np.array([text_image_length]*len(text))

        return text_image_output, image_init, text_init, cap_length


class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        # self.temperature = opt.temperature
        self.set_cuda = opt.cuda

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)  # 2 3 6
        )


    def forward(self, data_orgin, data_augment = None, labels=None, target_labels=None, text=None):
        orgin_res, image_init, text_init, text_length = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                     data_orgin.images, data_orgin.text_image_mask,text)

        # """grad_cam"""
        # orgin_res, image_init, text_init, text_length = self.fuse_model(data_orgin.texts,
        #                                                                 data_orgin.bert_attention_mask,
        #                                                                 data_orgin.images, data_orgin.text_image_mask,
        #                                                                 data_orgin.text)
        output = self.output_classify(orgin_res)
        return output, image_init, text_init, text_length


class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = FuseModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])