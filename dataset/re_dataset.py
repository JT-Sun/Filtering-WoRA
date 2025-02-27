import json
import os
import random
import math
import numpy as np
from random import randint, shuffle
from random import random as rand
from PIL import Image
from PIL import ImageFile

import torch
from torch.utils.data import Dataset, IterableDataset
from dataset.dist_dataset import DistLineReadingDataset
from dataset.utils import pre_caption
import itertools
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from train_tools import mlm_new
from models.tokenization_bert import BertTokenizer
import copy

class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        self.use_roberta = use_roberta
        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

        print("len(tokenizer.id2token): ", len(self.id2token), "  ----  cls_token_id: ", self.cls_token_id,
              "  ----  mask_token_id: ", self.mask_token_id, flush=True)

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return i  # self.id2token[i]

    def __call__(self, text_ids):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(text_ids) * self.mask_prob))))

        # candidate positions of masked tokens
        assert text_ids[0] == self.cls_token_id
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(text_ids)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (self.id2token[text_ids[new_st].item()][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and (self.id2token[text_ids[new_end].item()][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and self.id2token[text_ids[new_st].item()].startswith('##'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and self.id2token[text_ids[new_end].item()].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                text_ids[pos] = self.mask_token_id
            elif rand() < 0.5:  # 10%
                text_ids[pos] = self.get_random_word()

        return text_ids, masked_pos
class TextMaskingGenerator_box:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True, use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token
        print("mask_generator.cls_token, ", self.cls_token, flush=True)
        print("mask_generator.mask_token, ", self.mask_token, flush=True)

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos

class re_train_dataset(Dataset):
    def __init__(self, config, transform, pre_transform):
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']
        self.eda = config['eda']
        self.eda_p = config['eda_p']
        ann_file = config['train_file']

        if ('attr' in config.keys()) and config['attr']:
            self.attr = True
        else:
            self.attr = False

        self.transform = transform
        self.pre_transform = pre_transform
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.img_ids = {}

        n = 1
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        try:
            image_path = os.path.join(self.image_root, ann['image'])
        except:
            print("self.image_root", self.image_root)
            print("ann['image']", ann['image'])
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)
        if self.eda:
            caption1 = pre_caption(ann['caption'], self.max_words, self.icfg_rstp, True, self.eda_p)
            return image1, caption, caption1, self.img_ids[ann['image_id']]
        elif self.attr:
            label = torch.tensor(ann['label'])
            return image1, caption, self.img_ids[ann['image_id']], label
        else:
            return image1, caption, self.img_ids[ann['image_id']]
# class re_train_dataset(Dataset):
#     def __init__(self, config, transform, pre_transform):
#         self.image_root = config['image_root']
#         self.max_words = config['max_words']
#         self.icfg_rstp = config['icfg_rstp']
#         self.eda = config['eda']
#         self.eda_p = config['eda_p']
#         self.ann_files = config['train_file']
#
#         self.attr = config.get('attr', False)
#         self.box = config.get('box', False)
#         self.transform = transform
#         self.pre_transform = pre_transform
#         self.ann_loaded = False
#
#     def load_annotations(self):
#         if not self.ann_loaded:
#             self.ann = []
#             for f in self.ann_files:
#                 with open(f, 'r') as file:
#                     self.ann += json.load(file)
#             self.ann_loaded = True
#
#     def get_img_id(self, image_id):
#         if not hasattr(self, 'img_ids'):
#             self.img_ids = {ann['image_id']: idx + 1 for idx, ann in enumerate(self.ann)}
#         return self.img_ids[image_id]
#
#     def __len__(self):
#         self.load_annotations()
#         return len(self.ann)
#
#     def __getitem__(self, index):
#         self.load_annotations()
#         ann = self.ann[index]
#
#         image_path = os.path.join(self.image_root, ann['image'])
#         image = Image.open(image_path).convert('RGB')
#         image1 = self.transform(image) if self.transform else image
#
#         caption = pre_caption(ann['caption'], self.max_words)
#         if self.eda:
#             caption1 = pre_caption(ann['caption'], self.max_words, self.icfg_rstp, True, self.eda_p)
#             return image1, caption, caption1, self.get_img_id(ann['image_id'])
#         elif self.attr:
#             label = torch.tensor(ann['label'])
#             # if self.box:
#             #     target_bbox = torch.tensor(ann['boxes'], dtype=torch.float32)
#             #     logits = torch.tensor(ann['logits'], dtype=torch.float32)
#             return image1, caption, self.get_img_id(ann['image_id']), label
#             # else:
#             #     return image1, caption, self.get_img_id(ann['image_id']), label
#         else:
#             return image1, caption, self.get_img_id(ann['image_id'])

# class DistributedIterableDataset(IterableDataset):
#     def __init__(self, dataset, num_replicas=None, rank=None):
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#
#
#     def __iter__(self):
#         # 为每个进程生成一个迭代器
#         iterator = iter(self.dataset)
#         # 跳过不属于当前进程的元素
#         iterator = itertools.islice(iterator, self.rank, None, self.num_replicas)
#         # 可以在此处进行shuffle，但请注意shuffle在IterableDataset中的实现可能会有内存限制
#         return iterator

class re_train_dataset_iter(DistLineReadingDataset):
    def __init__(self, config, transform, pre_transform, data_path, rank=0, world_size=4, shuffle=True, repeat=True):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']
        self.eda = config['eda']
        self.eda_p = config['eda_p']
        ann_file = config['train_file']

        if ('attr' in config.keys()) and config['attr']:
            self.attr = True
        else:
            self.attr = False
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.transform = transform
        self.pre_transform = pre_transform
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.img_ids = {}

        self.tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
        self.max_tokens = config['max_tokens']
        n = 1
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __iter__(self):
        # # 计算每个进程应处理的数据数量
        # per_process_data_len = math.ceil(len(self.ann) / self.world_size)
        #
        # # 计算当前进程的数据起始和结束索引
        # start_idx = self.rank * per_process_data_len
        # end_idx = min(start_idx + per_process_data_len, len(self.ann))
        # # 如果启用了shuffle，对当前进程的数据索引进行洗牌
        # if self.shuffle:
        #     indices = list(range(start_idx, end_idx))
        #     random.shuffle(indices)
        # else:
        #     indices = range(start_idx, end_idx)
        # # 仅迭代当前进程负责的数据部分
        # for idx in indices:
        #     ann = self.ann[idx]
        #     try:
        for example in self.generate():
            try:
        # for ann in self.ann:
        #     try:
                ann = example
                image_path = os.path.join(self.image_root, ann['image'])
                image = Image.open(image_path).convert('RGB')
                image1 = self.transform(image)

                caption = pre_caption(ann['caption'], self.max_words)
                if self.eda:
                    caption1 = pre_caption(ann['caption'], self.max_words, self.icfg_rstp, True, self.eda_p)
                    yield image1, caption, caption1, self.img_ids[ann['image_id']]
                elif self.attr:
                    label = torch.tensor(ann['label'])
                    yield image1, caption, self.img_ids[ann['image_id']], label
                else:
                    yield image1, caption, self.img_ids[ann['image_id']]
            except Exception as e:
                print("Error processing image:", ann['image'], "\nException:", e)
    # def collate_fn(self, batch):
    #     batch_tensors = []
    #     for x in zip(*batch):
    #         if x[0] is None:
    #             batch_tensors.append(None)
    #         elif isinstance(x[0], torch.Tensor):
    #             batch_tensors.append(torch.stack(x))
    #         else:
    #             batch_tensors.append(torch.tensor(x, dtype=torch.long))
    #
    #     return batch_tensors
class re_train_dataset_addbox(Dataset):
    def __init__(self, config, transform, pre_transform):
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']
        self.eda = config['eda']
        self.eda_p = config['eda_p']
        self.ann_files = config['train_file']

        self.attr = config.get('attr', False)
        self.box = config.get('box', False)
        self.transform = transform
        self.pre_transform = pre_transform
        self.ann_loaded = False
        # self.max_tokensconfig['max_tokens']
        self.tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

        self.mask_generator_box = TextMaskingGenerator_box(self.tokenizer, config['mask_prob'], config['max_masks'],
                                                  config['skipgram_prb'], config['skipgram_size'],
                                                  config['mask_whole_word'])

        self.cls_token = self.tokenizer.cls_token
        # self.eos_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_tokens = config['max_tokens']
        self.max_masks = config['max_masks']
        self.tokenized = False
        self.config = config
        # 根据需要调整 patch_size 和 num_patch
        self.patch_size = config['patch_size']
        self.num_patch_x = int(config['w']/config['patch_size'])
        self.num_patch_y = int(config['h']/config['patch_size'])
        # self.patch_size = config['patch_size']
        # assert self.image_res % self.patch_size == 0
        # self.num_patch = int(self.image_res / self.patch_size)
    def load_annotations(self):
        if not self.ann_loaded:
            self.ann = []
            for f in self.ann_files:
                with open(f, 'r') as file:
                    self.ann += json.load(file)
            self.ann_loaded = True

    def get_img_id(self, image_id):
        if not hasattr(self, 'img_ids'):
            self.img_ids = {ann['image_id']: idx + 1 for idx, ann in enumerate(self.ann)}
        return self.img_ids[image_id]

    def __len__(self):
        self.load_annotations()
        return len(self.ann)

    def __getitem__(self, index):
        self.load_annotations()
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        original_image = Image.open(image_path).convert('RGB')
        W, H = original_image.size
        image1 = self.transform(original_image) if self.transform else original_image

        # text_data_list = []
        target_bbox_list = []
        logits_list = []
        idx_to_group_img = []
        image_atts_list = []

        text_ids_list = []
        text_ids_masked_list = []
        text_atts_list = []
        masked_pos_list = []
        masked_ids_list = []
        is_image_list = []

        # 处理总 caption
        caption = pre_caption(ann['caption'], self.max_words)
        text_input = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_tokens,return_tensors="pt")
        text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

        text_ids_list.append(text_ids)
        text_atts_list.append(text_atts)
        text_ids_masked_list.append(text_ids_masked)
        masked_pos_list.append(masked_pos)
        masked_ids_list.append(masked_ids)
        # text_data_list.append(caption)
        image_atts_list.append([1] * (self.num_patch_x * self.num_patch_y + 1))
        target_bbox_list.append(torch.tensor([0.5, 0.5, 1, 1], dtype=torch.float))
        # print(f"Before processing boxes: {len(image_atts_list)} attention maps for image {index}")

        is_image_list.append(1)
        # if self.box:
        # 为每个 box 处理对应的图片和 caption
        for idx, elem in enumerate(ann['boxes']):
            # 裁剪图像
            x, y, w, h = elem
            # cropped_image = original_image.crop((x * W, y * H, (x + w) * W, (y + h) * H))
            # cropped_image = self.transform(cropped_image) if self.transform else cropped_image
            # image_list.append(cropped_image)
            # 添加 target_bbox
            # print(f"Processing box {idx} for image {index}")
            x_pixel = x * self.config['w']
            y_pixel = y * self.config['h']
            w_pixel = w * self.config['w']
            h_pixel = h * self.config['h']
            image_atts = self.get_image_attns(x_pixel, y_pixel, w_pixel, h_pixel)
            image_atts_list.append(image_atts)
            target_bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
            target_bbox_list.append(target_bbox)
            # # 每个 box 对应当前的图像索引
            # idx_to_group_img.append(index)
            # 处理和添加 phrase caption (如果存在)
            if 'phrases' in ann and idx < len(ann['phrases']):
                phrase_caption = ann['phrases'][idx]
                # 这里添加预处理后的 phrase caption
                phrase_text_ids, phrase_text_atts, phrase_text_ids_masked, phrase_masked_pos, phrase_masked_ids = self.preprocess(phrase_caption)
                # text_data_list.append(phrase_caption)
                text_ids_list.append(phrase_text_ids)
                text_atts_list.append(phrase_text_atts)
                text_ids_masked_list.append(phrase_text_ids_masked)
                masked_pos_list.append(phrase_masked_pos)
                masked_ids_list.append(phrase_masked_ids)
            if 'logits' in ann:
                logits = torch.tensor(ann['logits'][idx], dtype=torch.float32)
                logits_list.append(logits)
        # 如果同时使用 attr 和 box
        if self.attr:
            label = torch.tensor(ann['label'])
            # idx_to_group_img_tensor = torch.tensor(idx_to_group_img, dtype=torch.long)  # 将列表转换为张量
            # print(f"After processing boxes: {len(image_atts_list)} attention maps for image {index}")
            return image1, text_input.input_ids, text_input.attention_mask, self.get_img_id(ann['image_id']), label, text_ids_list, text_atts_list, text_ids_masked_list, masked_pos_list, \
                  masked_ids_list, target_bbox_list, logits_list, image_atts_list
        # elif self.attr:
        #     label = torch.tensor(ann['label'])
        #     yield image1, caption, self.get_img_id(ann['image_id']), label
        #
        # elif self.eda:
        #     # EDA 特定的逻辑
        #     caption1 = pre_caption(ann['caption'], self.max_words, self.icfg_rstp, True, self.eda_p)
        #     yield image1, caption, caption1, self.get_img_id(ann['image_id'])
        #
        # else:
        #     yield image1, caption, self.get_img_id(ann['image_id'])

    def get_image_attns(self, x, y, w, h):

        x_min = min(math.floor(x / self.patch_size), self.num_patch_x - 1)
        x_max = min(math.ceil((x + w) / self.patch_size), self.num_patch_x)

        y_min = min(math.floor(y / self.patch_size), self.num_patch_y - 1)
        y_max = min(math.ceil((y + h) / self.patch_size), self.num_patch_y)

        image_atts = [0] * (1 + self.num_patch_x * self.num_patch_y)
        image_atts[0] = 1  # always include [CLS]
        for j in range(y_min, y_max):
            for i in range(x_min, x_max):
                index = self.num_patch_x * j + i + 1
                assert (index > 0) and (index <= self.num_patch_x * self.num_patch_y), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts

    def preprocess(self, text):
        if self.tokenized:
            tokens = text.strip().split(' ')
        else:
            text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
            tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        # if self.add_eos:
        #     tokens = tokens[:self.max_tokens - 1]
        #     tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int
        tokens_masked, masked_pos = self.mask_generator_box(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]
        # 使用 tokenizer 和 mask_generator 生成 masked 文本
        # text_input = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=self.max_tokens,
        #                             return_tensors="pt")
        # text_ids_masked, masked_pos, masked_ids = mlm_new(tokens, text_input, self.tokenizer, self.mask_generator, self.config)

        # pad 填充文本以达到 max_tokens
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        # 填充 masked 文本以达到 max_masks
        # n_pad_masked = self.max_masks - len(masked_ids[0])
        # masked_pos = [p + [0] * n_pad_masked for p in masked_pos]
        # masked_ids = [m + [-100] * n_pad_masked for m in masked_ids]
        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        # n_pad_masked = self.max_masks - len(masked_ids[0])
        # masked_pos = [p.tolist() + [0] * n_pad_masked for p in masked_pos]  # 确保 p 是列表
        # masked_ids = [m.tolist() + [-100] * n_pad_masked for m in masked_ids]  # 将张量转换为列表并添加填充
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [-100] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

class re_train_dataset_addbox_iter(re_train_dataset_iter):
    def __init__(self, config, transform, pre_transform, data_path, rank=0, world_size=4, shuffle=True, repeat=True):
        super().__init__(config, transform, pre_transform, data_path, rank, world_size, shuffle, repeat)
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']
        self.eda = config['eda']
        self.eda_p = config['eda_p']
        self.ann_files = config['train_file']

        self.attr = config.get('attr', False)
        self.box = config.get('box', False)
        self.transform = transform
        self.pre_transform = pre_transform
        self.ann_loaded = False
        # self.max_tokensconfig['max_tokens']
        self.tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

        self.mask_generator_box = TextMaskingGenerator_box(self.tokenizer, config['mask_prob'], config['max_masks'],
                                                  config['skipgram_prb'], config['skipgram_size'],
                                                  config['mask_whole_word'])

        self.cls_token = self.tokenizer.cls_token
        # self.eos_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_tokens = config['max_tokens']
        self.max_masks = config['max_masks']
        self.tokenized = False
        self.config = config
        # 根据需要调整 patch_size 和 num_patch
        self.patch_size = config['patch_size']
        self.num_patch_x = int(config['w']/config['patch_size'])
        self.num_patch_y = int(config['h']/config['patch_size'])
        # self.patch_size = config['patch_size']
        # assert self.image_res % self.patch_size == 0
        # self.num_patch = int(self.image_res / self.patch_size)
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.max_regions = config['regions']['max_regions']
        self.batch_size = config['regions']['batch_size']
    def load_annotations(self):
        if not self.ann_loaded:
            self.ann = []
            for f in self.ann_files:
                with open(f, 'r') as file:
                    self.ann += json.load(file)
            self.ann_loaded = True

    def get_img_id(self, image_id):
        if not hasattr(self, 'img_ids'):
            self.img_ids = {ann['image_id']: idx + 1 for idx, ann in enumerate(self.ann)}
        return self.img_ids[image_id]

    # def __len__(self):
    #     self.load_annotations()
    #     return len(self.ann)

    def __iter__(self):
        # self.load_annotations()
        # # # 计算每个进程应处理的数据数量
        # # per_process_data_len = math.ceil(len(self.ann) / self.world_size)
        # #
        # # # 计算当前进程的数据起始和结束索引
        # # start_idx = self.rank * per_process_data_len
        # # end_idx = min(start_idx + per_process_data_len, len(self.ann))
        # #
        # # # 创建索引列表
        # # indices = list(range(start_idx, end_idx))
        # #
        # # # 如果启用了 shuffle，对索引进行洗牌
        # # if self.shuffle:
        # #     random.shuffle(indices)
        # #
        # # # 仅迭代当前进程负责的数据部分
        # # for idx in indices:
        # #     ann = self.ann[idx]
        # for ann in self.ann:
        for example in self.generate():
            try:
                ann = example
                image_path = os.path.join(self.image_root, ann['image'])
                original_image = Image.open(image_path).convert('RGB')
                W, H = original_image.size
                image1 = self.transform(original_image) if self.transform else original_image

                # text_data_list = []
                target_bbox_list = []
                logits_list = []
                idx_to_group_img = []
                image_atts_list = []

                text_ids_list = []
                text_ids_masked_list = []
                text_atts_list = []
                masked_pos_list = []
                masked_ids_list = []
                is_image_list = []
                max_elems = self.max_regions
                # 处理总 caption

                caption = pre_caption(ann['caption'], self.max_words)
                text_input = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_tokens,return_tensors="pt")
                text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

                text_ids_list.append(text_ids)
                text_atts_list.append(text_atts)
                text_ids_masked_list.append(text_ids_masked)
                masked_pos_list.append(masked_pos)
                masked_ids_list.append(masked_ids)
                # text_data_list.append(caption)
                image_atts_list.append([1] * (self.num_patch_x * self.num_patch_y + 1))
                target_bbox_list.append(torch.tensor([0.5, 0.5, 1, 1], dtype=torch.float))
                logits_list.append(torch.tensor(1.0))
                # print(f"Before processing boxes: {len(image_atts_list)} attention maps for image {index}")
                max_elems -= 1
                # is_image_list.append(1)
                # if self.box:
                # 为每个 box 处理对应的图片和 caption
                for idx, elem in enumerate(ann['boxes']):
                    if max_elems <= 0:
                        break
                    # 裁剪图像
                    x, y, w, h = elem
                    # cropped_image = original_image.crop((x * W, y * H, (x + w) * W, (y + h) * H))
                    # cropped_image = self.transform(cropped_image) if self.transform else cropped_image
                    # image_list.append(cropped_image)
                    # 添加 target_bbox
                    # print(f"Processing box {idx} for image {index}")
                    x_pixel = x * self.config['w']
                    y_pixel = y * self.config['h']
                    w_pixel = w * self.config['w']
                    h_pixel = h * self.config['h']
                    image_atts = self.get_image_attns(x_pixel, y_pixel, w_pixel, h_pixel)
                    # image_atts = self.get_image_attns(x, y, w, h)
                    image_atts_list.append(image_atts)
                    target_bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
                    target_bbox_list.append(target_bbox)
                    # # 每个 box 对应当前的图像索引
                    # idx_to_group_img.append(index)
                    # 处理和添加 phrase caption (如果存在)
                    if 'phrases' in ann and idx < len(ann['phrases']):
                        phrase_caption = ann['phrases'][idx]
                        # 这里添加预处理后的 phrase caption
                        phrase_text_ids, phrase_text_atts, phrase_text_ids_masked, phrase_masked_pos, phrase_masked_ids = self.preprocess(phrase_caption)
                        # text_data_list.append(phrase_caption)
                        text_ids_list.append(phrase_text_ids)
                        text_atts_list.append(phrase_text_atts)
                        text_ids_masked_list.append(phrase_text_ids_masked)
                        masked_pos_list.append(phrase_masked_pos)
                        masked_ids_list.append(phrase_masked_ids)
                    if 'logits' in ann :
                        logits = torch.tensor(ann['logits'][idx], dtype=torch.float32)
                        logits_list.append(logits)
                    max_elems -= 1
                # 如果同时使用 attr 和 box
                image_list = [image1] if len(text_ids_list) else []
                if self.attr:
                    label = torch.tensor(ann['label'])
                    # idx_to_group_img_tensor = torch.tensor(idx_to_group_img, dtype=torch.long)  # 将列表转换为张量
                    # print(f"After processing boxes: {len(image_atts_list)} attention maps for image {index}")
                    yield image_list, image1, text_input.input_ids, text_input.attention_mask, self.get_img_id(ann['image_id']), label, text_ids_list, text_atts_list, text_ids_masked_list, masked_pos_list, \
                          masked_ids_list, target_bbox_list, logits_list, image_atts_list
            except Exception as e:
                print("Error processing image:", ann['image'], "\nException:", e)
            # elif self.attr:
            #     label = torch.tensor(ann['label'])
            #     yield image1, caption, self.get_img_id(ann['image_id']), label
            #
            # elif self.eda:
            #     # EDA 特定的逻辑
            #     caption1 = pre_caption(ann['caption'], self.max_words, self.icfg_rstp, True, self.eda_p)
            #     yield image1, caption, caption1, self.get_img_id(ann['image_id'])
            #
            # else:
            #     yield image1, caption, self.get_img_id(ann['image_id'])

    def get_image_attns(self, x, y, w, h):

        x_min = min(math.floor(x / self.patch_size), self.num_patch_x - 1)
        x_max = min(math.ceil((x + w) / self.patch_size), self.num_patch_x)

        y_min = min(math.floor(y / self.patch_size), self.num_patch_y - 1)
        y_max = min(math.ceil((y + h) / self.patch_size), self.num_patch_y)

        image_atts = [0] * (1 + self.num_patch_x * self.num_patch_y)
        image_atts[0] = 1  # always include [CLS]
        for j in range(y_min, y_max):
            for i in range(x_min, x_max):
                index = self.num_patch_x * j + i + 1
                assert (index > 0) and (index <= self.num_patch_x * self.num_patch_y), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts

    def preprocess(self, text):
        if self.tokenized:
            tokens = text.strip().split(' ')
        else:
            text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
            tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        # if self.add_eos:
        #     tokens = tokens[:self.max_tokens - 1]
        #     tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int
        tokens_masked, masked_pos = self.mask_generator_box(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]
        # 使用 tokenizer 和 mask_generator 生成 masked 文本
        # text_input = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=self.max_tokens,
        #                             return_tensors="pt")
        # text_ids_masked, masked_pos, masked_ids = mlm_new(tokens, text_input, self.tokenizer, self.mask_generator, self.config)

        # pad 填充文本以达到 max_tokens
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        # 填充 masked 文本以达到 max_masks
        # n_pad_masked = self.max_masks - len(masked_ids[0])
        # masked_pos = [p + [0] * n_pad_masked for p in masked_pos]
        # masked_ids = [m + [-100] * n_pad_masked for m in masked_ids]
        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        # n_pad_masked = self.max_masks - len(masked_ids[0])
        # masked_pos = [p.tolist() + [0] * n_pad_masked for p in masked_pos]  # 确保 p 是列表
        # masked_ids = [m.tolist() + [-100] * n_pad_masked for m in masked_ids]  # 将张量转换为列表并添加填充
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [-100] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

    def collate_fn(self, batch_sample):
        batch = list(zip(*batch_sample))

        # 解包批量数据
        image_list, images, text_ids, text_attmask, image_ids, labels, batch = batch[0], batch[1], batch[2], batch[3], batch[
            4], batch[5], batch[6:]

        # 堆叠或转换基本的批量数据
        images = torch.stack(images)
        text_ids = torch.stack(text_ids) if text_ids[0].dim() != 0 else torch.tensor(text_ids, dtype=torch.long)
        text_attmask = torch.stack(text_attmask) if text_attmask[0].dim() != 0 else torch.tensor(text_attmask,
                                                                                                 dtype=torch.long)
        image_ids = torch.tensor(image_ids, dtype=torch.long)
        labels = torch.stack(labels) if labels[0].dim() != 0 else torch.tensor(labels, dtype=torch.long)

        # print(len(image_atts_list))
        idx_to_group_img = []
        img_idx = -1
        for sample in batch[0]:
            n_elems = len(sample)
            if n_elems > 0:
                img_idx += 1
                idx_to_group_img.extend([img_idx] * n_elems)  # flatten

        batch_size = self.batch_size
        n_elems = len(idx_to_group_img)
        to_keep = list(range(n_elems))
        if n_elems >= batch_size:
            to_keep = random.sample(to_keep, batch_size)
        else:
            # fixed batch_size is required. otherwise, the process will be blocked. so, i do pad here.
            # but pad causes wrong calculation for contrastive learning.
            # Set appropriate batch_size, max_images, and max_regions to avoid frequent padding.
            try:
                to_pad = random.sample(to_keep, batch_size - n_elems)
                to_keep += to_pad
                print("### warning: pad region_batch by sampling, ", len(to_pad), flush=True)

            except ValueError:
                print("### warning: pad region_batch by expanding, ", batch_size - len(to_keep), flush=True)
                to_keep = (to_keep * math.ceil(batch_size / len(to_keep)))[:batch_size]
        idx_to_group_img = torch.tensor([idx_to_group_img[index] for index in to_keep], dtype=torch.long)
        # 将所有数据添加到 batch_tensors
        batch_tensors = [images, text_ids, text_attmask, image_ids, labels, idx_to_group_img]
        for x in [sum(x, []) for x in batch]:

            x = [x[index] for index in to_keep]

            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors



class re_test_dataset(Dataset):
    def __init__(self, ann_file, config, transform):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.g_pids = []
        self.q_pids = []

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.g_pids.append(ann['image_id'])
            self.image.append(ann['image'])
            self.img2txt[img_id] = []

            t = 0
            for i, caption in enumerate(ann['caption']):
                self.q_pids.append(ann['image_id'])
                self.text.append(pre_caption(caption, self.max_words, icfg_rstp=self.icfg_rstp))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = []
                self.txt2img[txt_id].append(img_id)
                txt_id += 1
                t += 1

            txt_id1 = 0
            for img_id1, ann1 in enumerate(self.ann):
                for i1, caption1 in enumerate(ann1['caption']):
                    if ann['image_id'] == ann1['image_id'] and img_id != img_id1:
                        self.img2txt[img_id].append(txt_id1)
                    txt_id1 += 1
                if ann['image_id'] == ann1['image_id'] and img_id != img_id1:
                    for temp in range(t):
                        self.txt2img[txt_id - 1 - temp].append(img_id1)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class re_test_dataset_icfg(Dataset):
    def __init__(self, config, transform):
        ann_file = config['test_file']
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.g_pids = []
        self.q_pids = []

        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.g_pids.append(ann['image_id'])
            self.img2txt[img_id] = []
            self.img2txt[img_id].append(img_id)

            self.text.append(pre_caption(ann['caption'][0], self.max_words, icfg_rstp=True))
            self.q_pids.append(ann['image_id'])

            self.txt2img[img_id] = []
            self.txt2img[img_id].append(img_id)

            for img_id1, ann1 in enumerate(self.ann):
                if ann['image_id'] == ann1['image_id'] and img_id != img_id1:
                    self.txt2img[img_id].append(img_id1)
                    self.img2txt[img_id].append(img_id1)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class re_train_dataset_attr(Dataset):
    def __init__(self, config, transform):
        ann_file = config['train_file']
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(ann['label'])
        return image, label


class re_test_dataset_attr(Dataset):
    def __init__(self, ann_file, config, transform):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']

        self.image = []
        self.label = []
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.label.append(ann['label'])
        self.label = np.array(self.label)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index
