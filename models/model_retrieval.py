import torch
from models import APTM, load_pretrained, AllGather
import torch.nn as nn
import torch.nn.functional as F


class APTM_Retrieval(APTM):
    def __init__(self, config):
        super().__init__(config, load_vision_params=config['load_params'], load_text_params=config['load_params'],
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm'], use_bbox_loss=config['box'])

        if not self.pa100k_only_img_classifier:
            self.mlm = config['mlm']
            self.pa100k = config['pa100k']
            if not self.pa100k:
                self.eda = config['eda']
            if ('attr' in config.keys()) and config['attr']:
                self.attr = True
            else:
                self.attr = False
            if ('box' in config.keys()) and config['box']:
                self.box = True
            else:
                self.box = False

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("vision_encoder missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' in p])
        print("unexpected_keys: ", msg.unexpected_keys)
        # 打印加载的所有模型参数的名称和形状
        # print("Loaded model parameters:")
        # for name, param in self.named_parameters():
        #     print(f"{name}: {param.size()}")

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, attr_text_ids=None, attr_text_atts=None, attr_text_ids_masked=None,
                attr_masked_pos=None, attr_masked_ids=None, label=None, text_ids_eda=None, text_atts_eda=None):

        if self.pa100k_only_img_classifier:
            image_embeds = self.vision_encoder(image)
            outputs = self.img_cls(image_embeds[:, 0, :])
            loss = self.criterion(outputs, label.float())
            return loss

        if self.pa100k:
            image_embeds, image_atts = self.get_vision_embeds(image)
            text_embeds = self.get_text_embeds(text_ids, text_atts)
            image_feat, text_feat = self.get_features(image_embeds, text_embeds)
            loss_itc = self.get_contrastive_loss_attr(image_feat, text_feat, label)
            loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, text_embeds, text_atts, label)
            if self.mlm:
                loss_mlm = self.get_mlm_loss_attr(text_ids_masked, text_atts, image_embeds, image_atts,
                                                  masked_pos, masked_ids, label)
                return loss_itc, loss_itm, loss_mlm
            else:
                return loss_itc, loss_itm

        if self.attr:
            image_embeds, image_atts = self.get_vision_embeds(image)

            text_embeds = self.get_text_embeds(text_ids, text_atts)
            image_feat, text_feat = self.get_features(image_embeds, text_embeds)

            attr_text_embeds = self.get_text_embeds(attr_text_ids, attr_text_atts)
            attr_text_feat = self.get_features(text_embeds=attr_text_embeds)

            attr_loss_itc = self.get_contrastive_loss_attr(image_feat, attr_text_feat, label)
            attr_loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, attr_text_embeds, attr_text_atts,
                                                        label)

            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
            loss_itm, accuracy_info = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                              text_embeds, text_atts, text_feat, idx=idx)

            if self.mlm:
                attr_loss_mlm = self.get_mlm_loss_attr(attr_text_ids_masked, attr_text_atts, image_embeds, image_atts,
                                                       attr_masked_pos, attr_masked_ids, label)
                loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                             masked_ids)
                loss_attr = (attr_loss_itc + attr_loss_itm + attr_loss_mlm) / 3

                # if self.box:
                #     # 初始化损失累积变量
                #     accumulated_loss_bbox = 0.0
                #     accumulated_loss_giou = 0.0
                #
                #     # 对每个边界框计算损失
                #     for i in range(target_bbox.shape[1]):
                #         if torch.all(target_bbox[:, i, :] == 0):
                #             continue  # 跳过全零的边界框
                #         output_coord = self.predict_bbox(image_embeds, text_embeds, text_atts)
                #         loss_bbox_i, loss_giou_i = self.get_bbox_loss_nologits(output_coord, target_bbox[:, i, :],
                #                                                                logits, is_image=None)
                #
                #         # 累加损失
                #         accumulated_loss_bbox += loss_bbox_i
                #         accumulated_loss_giou += loss_giou_i
                #
                #     # 计算平均损失
                #     loss_bbox_total = accumulated_loss_bbox / target_bbox.shape[1] / 100
                #     loss_giou_total = accumulated_loss_giou / target_bbox.shape[1]
                #
                #     return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info, loss_bbox_total, loss_giou_total
                # else:
                #     return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info
                # if self.box:
                #     accumulated_loss_bbox = 0.0
                #     accumulated_loss_giou = 0.0
                #
                #     output_coord = self.predict_bbox(image_embeds, text_embeds, text_atts)
                #
                #     # 计算非零边界框的数量
                #     non_zero_boxes = torch.any(torch.any(target_bbox != 0, dim=2), dim=1)
                #     num_non_zero_boxes = non_zero_boxes.sum().item()
                #
                #     # 对每个边界框计算损失
                #     for i in range(target_bbox.shape[1]):
                #         if not non_zero_boxes[i]:
                #             continue  # 跳过全零的边界框
                #         loss_bbox_i, loss_giou_i = self.get_bbox_loss_nologits(output_coord, target_bbox[:, i, :],
                #                                                                logits, is_image=None)
                #
                #         # 累加损失
                #         accumulated_loss_bbox += loss_bbox_i
                #         accumulated_loss_giou += loss_giou_i
                #
                #     # 计算平均损失
                #     loss_bbox_total = accumulated_loss_bbox / max(num_non_zero_boxes, 1)
                #     loss_giou_total = accumulated_loss_giou / max(num_non_zero_boxes, 1)
                #
                #     return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info, loss_bbox_total, loss_giou_total
                #
                # else:
                return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info
            else:
                loss_attr = (attr_loss_itc + attr_loss_itm) / 2
                return loss_itc, loss_itm, loss_attr, accuracy_info

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        # loss_itm= self.get_matching_loss(image_embeds, image_atts, image_feat,
        #                                   text_embeds, text_atts, text_feat, idx=idx)
        loss_itm, accuracy_info = self.get_matching_loss_acc(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)

        # eda
        if self.eda:
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_features(text_embeds=text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
            loss_itm_eda, accuracy_info_eda = self.get_matching_loss_acc(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx)
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda
            # accuracy_info = accuracy_info + 0.8 * accuracy_info_eda

        if self.mlm:
            loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                         masked_ids)

            return loss_itc, loss_itm, loss_mlm, accuracy_info
        else:
            return loss_itc, loss_itm, accuracy_info


class APTM_Retrieval_new(APTM):
    def __init__(self, config):
        super().__init__(config, load_vision_params=config['load_params'], load_text_params=config['load_params'],
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm'], use_bbox_loss=config['box'])

        if not self.pa100k_only_img_classifier:
            self.mlm = config['mlm']
            self.pa100k = config['pa100k']
            if not self.pa100k:
                self.eda = config['eda']
            if ('attr' in config.keys()) and config['attr']:
                self.attr = True
            else:
                self.attr = False
            if ('box' in config.keys()) and config['box']:
                self.box = True
            else:
                self.box = False

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("vision_encoder missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' in p])
        print("unexpected_keys: ", msg.unexpected_keys)
        # 打印加载的所有模型参数的名称和形状
        # print("Loaded model parameters:")
        # for name, param in self.named_parameters():
        #     print(f"{name}: {param.size()}")

    def forward(self, image, text_ids, text_atts, text_ids_new=None, text_atts_new=None, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, attr_text_ids=None, attr_text_atts=None, attr_text_ids_masked=None,
                attr_masked_pos=None, attr_masked_ids=None, label=None, text_ids_eda=None, text_atts_eda=None, target_bbox_list=None, logits_list=None, idx_to_group_img=None, image_atts=None, is_image=None):

        if self.pa100k_only_img_classifier:
            image_embeds = self.vision_encoder(image)
            outputs = self.img_cls(image_embeds[:, 0, :])
            loss = self.criterion(outputs, label.float())
            return loss

        if self.pa100k:
            image_embeds, image_atts = self.get_vision_embeds(image)
            text_embeds = self.get_text_embeds(text_ids, text_atts)
            image_feat, text_feat = self.get_features(image_embeds, text_embeds)
            loss_itc = self.get_contrastive_loss_attr(image_feat, text_feat, label)
            loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, text_embeds, text_atts, label)
            if self.mlm:
                loss_mlm = self.get_mlm_loss_attr(text_ids_masked, text_atts, image_embeds, image_atts,
                                                  masked_pos, masked_ids, label)
                return loss_itc, loss_itm, loss_mlm
            else:
                return loss_itc, loss_itm

        if image_atts is not None:
            image_embeds_new, image_atts_new, image_embeds_fullatts = \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
            image_embeds, image_atts1 = self.get_vision_embeds(image)
            if self.attr:
                text_embeds_new = self.get_text_embeds(text_ids_new, text_atts_new)
                text_ids = torch.squeeze(text_ids, dim=1)
                text_atts = torch.squeeze(text_ids, dim=1)
                text_embeds = self.get_text_embeds(text_ids, text_atts)

                image_feat_new, text_feat_new = self.get_features(image_embeds_new, text_embeds_new)
                image_feat, text_feat = self.get_features(image_embeds, text_embeds)

                attr_text_embeds = self.get_text_embeds(attr_text_ids, attr_text_atts)
                attr_text_feat = self.get_features(text_embeds=attr_text_embeds)

                attr_loss_itc = self.get_contrastive_loss_attr(image_feat, attr_text_feat, label)
                attr_loss_itm = self.get_matching_loss_attr(image_embeds, image_atts1, attr_text_embeds, attr_text_atts,
                                                            label)

                # loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
                loss_itc = self.get_contrastive_loss(image_feat_new, text_feat_new, idx=None)
                loss_itm, accuracy_info = self.get_matching_loss(image_embeds_new, image_atts_new, image_feat_new,
                                                  text_embeds_new, text_atts_new, text_feat_new, idx=None)

                if self.mlm:
                    attr_loss_mlm = self.get_mlm_loss_attr(attr_text_ids_masked, attr_text_atts, image_embeds, image_atts,
                                                           attr_masked_pos, attr_masked_ids, label)
                    loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts_new, image_embeds_new, image_atts_new, masked_pos,
                                                 masked_ids)
                    loss_attr = (attr_loss_itc + attr_loss_itm + attr_loss_mlm) / 3

                    # if self.box:
                    #     # 初始化损失累积变量
                    #     accumulated_loss_bbox = 0.0
                    #     accumulated_loss_giou = 0.0
                    #
                    #     # 对每个边界框计算损失
                    #     for i in range(target_bbox.shape[1]):
                    #         if torch.all(target_bbox[:, i, :] == 0):
                    #             continue  # 跳过全零的边界框
                    #         output_coord = self.predict_bbox(image_embeds, text_embeds, text_atts)
                    #         loss_bbox_i, loss_giou_i = self.get_bbox_loss_nologits(output_coord, target_bbox[:, i, :],
                    #                                                                logits, is_image=None)
                    #
                    #         # 累加损失
                    #         accumulated_loss_bbox += loss_bbox_i
                    #         accumulated_loss_giou += loss_giou_i
                    #
                    #     # 计算平均损失
                    #     loss_bbox_total = accumulated_loss_bbox / target_bbox.shape[1] / 100
                    #     loss_giou_total = accumulated_loss_giou / target_bbox.shape[1]
                    #
                    #     return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info, loss_bbox_total, loss_giou_total
                    # else:
                    #     return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info

                        # accumulated_loss_bbox = 0.0
                        # accumulated_loss_giou = 0.0
                    output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds_new, text_atts_new)
                    loss_bbox, loss_giou = self.get_bbox_loss_nologits(output_coord, target_bbox_list, logits_list,is_image=is_image)
                        # # 计算非零边界框的数量
                        # non_zero_boxes = torch.any(torch.any(target_bbox != 0, dim=2), dim=1)
                        # num_non_zero_boxes = non_zero_boxes.sum().item()
                        #
                        # # 对每个边界框计算损失
                        # for i in range(target_bbox.shape[1]):
                        #     if not non_zero_boxes[i]:
                        #         continue  # 跳过全零的边界框
                        #     loss_bbox_i, loss_giou_i = self.get_bbox_loss_nologits(output_coord, target_bbox[:, i, :],
                        #                                                            logits, is_image=is_image)
                        #
                        #     # 累加损失
                        #     accumulated_loss_bbox += loss_bbox_i
                        #     accumulated_loss_giou += loss_giou_i
                        #
                        # # 计算平均损失
                        # loss_bbox_total = accumulated_loss_bbox / max(num_non_zero_boxes, 1)
                        # loss_giou_total = accumulated_loss_giou / max(num_non_zero_boxes, 1)
                        #
                        # return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info, loss_bbox_total, loss_giou_total
                    return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info, loss_bbox, loss_giou
                else:
                    loss_attr = (attr_loss_itc + attr_loss_itm) / 2
                    return loss_itc, loss_itm, loss_attr, accuracy_info
        else:
            if self.attr:
                image_embeds, image_atts = self.get_vision_embeds(image)
                text_embeds = self.get_text_embeds(text_ids, text_atts)
                image_feat, text_feat = self.get_features(image_embeds, text_embeds)

                attr_text_embeds = self.get_text_embeds(attr_text_ids, attr_text_atts)
                attr_text_feat = self.get_features(text_embeds=attr_text_embeds)

                attr_loss_itc = self.get_contrastive_loss_attr(image_feat, attr_text_feat, label)
                attr_loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, attr_text_embeds, attr_text_atts,
                                                            label)

                loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
                loss_itm, accuracy_info = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds, text_atts, text_feat, idx=idx)

                if self.mlm:
                    attr_loss_mlm = self.get_mlm_loss_attr(attr_text_ids_masked, attr_text_atts, image_embeds,
                                                           image_atts,
                                                           attr_masked_pos, attr_masked_ids, label)
                    loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                                 masked_ids)
                    loss_attr = (attr_loss_itc + attr_loss_itm + attr_loss_mlm) / 3
                    return loss_itc, loss_itm, loss_mlm, loss_attr, accuracy_info
                else:
                    loss_attr = (attr_loss_itc + attr_loss_itm) / 2
                    return loss_itc, loss_itm, loss_attr, accuracy_info
            else:
                image_embeds, image_atts = self.get_vision_embeds(image)
                text_embeds = self.get_text_embeds(text_ids, text_atts)
                image_feat, text_feat = self.get_features(image_embeds, text_embeds)
                loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
                # loss_itm= self.get_matching_loss(image_embeds, image_atts, image_feat,
                #                                   text_embeds, text_atts, text_feat, idx=idx)
                loss_itm, accuracy_info = self.get_matching_loss_acc(image_embeds, image_atts, image_feat,
                                                  text_embeds, text_atts, text_feat, idx=idx)

                # eda
                if self.eda:
                    text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
                    text_feat_eda = self.get_features(text_embeds=text_embeds_eda)
                    loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
                    loss_itm_eda, accuracy_info_eda = self.get_matching_loss_acc(image_embeds, image_atts, image_feat,
                                                          text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx)
                    loss_itc = loss_itc + 0.8 * loss_itc_eda
                    loss_itm = loss_itm + 0.8 * loss_itm_eda
                    # accuracy_info = accuracy_info + 0.8 * accuracy_info_eda

                if self.mlm:
                    loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                                 masked_ids)

                    return loss_itc, loss_itm, loss_mlm, accuracy_info
                else:
                    return loss_itc, loss_itm, accuracy_info
