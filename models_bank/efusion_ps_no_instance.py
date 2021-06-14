
import torch
from torch import nn
import torch.nn.functional as F
from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.continuous_conv import ContinuousConvolution
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
from common_blocks.image_list import ImageList
from segmentation_heads.RPN import RPN
from segmentation_heads.roi_heads import roi_heads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict
from utils.tensorize_batch import tensorize_batch
import temp_variables
import config_kitti
import matplotlib.pyplot as plt


class Two_D_Branch(nn.Module):
    def __init__(self, backbone_out_channels):
        super(Two_D_Branch, self).__init__()

        self.conv1 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(backbone_out_channels)
        )

        self.conv2 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(backbone_out_channels)
        )

        self.conv3 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(backbone_out_channels)
        )

    def forward(self, features):

        original_shape = features.shape[2:]
        conv1_out = F.relu(self.conv1(features))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = F.interpolate(conv2_out, original_shape)
        conv3_out = F.relu(self.conv3(features))

        return conv2_out + conv3_out


class Three_D_Branch(nn.Module):
    def __init__(self, n_feat, k_number, n_number=None):
        super(Three_D_Branch, self).__init__()

        self.branch_3d_continuous = nn.Sequential(
            ContinuousConvolution(n_feat, k_number, n_number),
            ContinuousConvolution(n_feat, k_number, n_number)
        )

    def forward(self, feats, mask, coors, indices):
        """
        mask: B x H x W
        feats: B x C x H x W
        coors: B x N x 3 (points coordinates)
        indices: B x N x K (knn indices, aka. mask_knn)
        """
        
        B, C, _, _ = feats.shape
        feats_mask = feats.permute(0, 2, 3, 1)[mask].view(B, -1, C)
        br_3d, _, _ = self.branch_3d_continuous(
            (feats_mask, coors, indices))  # B x N x C
        br_3d = br_3d.view(-1, C)  # B*N x C

        out = torch.zeros_like(feats.permute(0, 2, 3, 1))  # B x H x W x C
        out[mask] = br_3d
        out = out.permute(0, 3, 1, 2)  # B x C x H x W

        return out


class FuseBlock(nn.Module):
    def __init__(self, nin, nout, k_number, n_number=None, extra_output_layer=False):
        super(FuseBlock, self).__init__()

        self.extra_output_layer = extra_output_layer
        self.branch_2d = Two_D_Branch(nin)

        self.branch_3d = Three_D_Branch(nin, k_number, n_number)

        self.output_layer = nn.Sequential(
            # depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1, padding=1),
            depth_wise_sep_conv(nin, nout, kernel_size=3, padding=1),
            nn.BatchNorm2d(nout)
        )

    def forward(self, *inputs):

        # mask: B x H x W
        # feats: B x C x H x W
        # coors: B x N x 3 (points coordinates)
        # indices: B x N x K (knn indices, aka. mask_knn)

        feats, mask, coors, k_nn_indices = inputs[0]
        y = self.branch_3d(feats, mask, coors, k_nn_indices) + \
            self.branch_2d(feats)

        y = F.relu(self.output_layer(y))

        if self.extra_output_layer:
            y = y + feats
            return (y, mask, coors, k_nn_indices)

        return (y, mask, coors, k_nn_indices)


class EfusionPS(nn.Module):
    def __init__(self, k_number, num_ins_classes, num_sem_classes, original_image_size, n_number=None, min_size=640, max_size=1024, image_mean=None, image_std=None):
        super(EfusionPS, self).__init__()

        self.original_image_size = original_image_size

        self.sparse_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.rgbd_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # ----------1-------------
        self.fuse_conv_1 = nn.Sequential(
            FuseBlock(48, 64, k_number, n_number=n_number),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),  # 1
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True)  # 2
        )

        self.pool_conv_2x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )
        # -----------2----------------
        self.fuse_conv_2 = FuseBlock(
            64, 64, k_number, n_number=n_number, extra_output_layer=True)  # 3

        self.pool_conv_4x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # ------------3----------

        self.fuse_conv_3 = FuseBlock(
            64, 64, k_number, n_number=n_number, extra_output_layer=True)  # 4

        self.pool_conv_8x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # --------------4------------------

        self.fuse_conv_4 = nn.Sequential(
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True)
        )  # 5,6

        self.pool_conv_16x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # --------------5------------------

        self.fuse_conv_5 = nn.Sequential(
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True)
        )  # 7,8,9

        self.pool_conv_32x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.P4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        self.P8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        self.P16 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        self.P32 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        # -----------------Semantic---------------------
        self.semantic_head = sem_seg_head(
            256, num_ins_classes + num_sem_classes + 1, original_image_size)
        # ----------------Instance-----------------------------------
        # self.rpn = RPN(256)
        # self.roi_pool = roi_heads(num_ins_classes + 1)

        # if image_mean is None:
        #     image_mean = [0.485, 0.456, 0.406]
        # if image_std is None:
        #     image_std = [0.229, 0.224, 0.225]

        # self.transform = GeneralizedRCNNTransform(
        #     min_size, max_size, image_mean, image_std)
        # -----------------------------------------

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, img, sparse_depth, mask, coors, k_nn_indices,  anns=None, sparse_depth_gt=None):
        """
        inputs:
        img: input rgb (B x 3 x H x W)
        sparse_depth: input sparse depth (B x 1 x H x W)
        coors: sparse 3D points (B x 3 x N)
        mask: mask_2d3d (B x H x W)
        indices: mask_knn (B x N x K)

        output:
        depth: completed depth
        """

        _, H, W = mask.shape
        losses = {}
        # sparse depth branch
        y_sparse = self.sparse_conv(sparse_depth)  # B x 16 x H/2 x W/2

        # rgbd branch
        x_concat_d = torch.cat((img, sparse_depth), dim=1)
        y_rgbd = self.rgbd_conv(x_concat_d)  # B x 32 x H/2 x W/2

        y_rgbd_concat_y_sparse = torch.cat((y_rgbd, y_sparse), dim=1)

        y_rgbd_concat_y_sparse = F.interpolate(y_rgbd_concat_y_sparse, (H, W))

        # -----------------------------------------
        
        fused_1, _, _, _ = self.fuse_conv_1(
            (y_rgbd_concat_y_sparse, mask, coors, k_nn_indices))
        # out_2x = self.pool_conv_2x(fused_1)

        fused_2, _, _, _ = self.fuse_conv_2(
            (fused_1, mask, coors, k_nn_indices))
        out_4x = self.pool_conv_4x(fused_2)

        fused_3, _, _, _ = self.fuse_conv_3(
            (fused_2, mask, coors, k_nn_indices))
        out_8x = self.pool_conv_8x(fused_3)

        fused_4, _, _, _ = self.fuse_conv_4(
            (fused_3, mask, coors, k_nn_indices))
        out_16x = self.pool_conv_16x(fused_4)

        fused_5, _, _, _ = self.fuse_conv_5(
            (fused_4, mask, coors, k_nn_indices))
        out_32x = self.pool_conv_32x(fused_5)

        # ---------------2 way FPN------------------

        # BOTTOM UP
        b_up1 = out_32x
        b_up2 = F.interpolate(b_up1, size=out_16x.shape[2:]) + out_16x
        b_up3 = F.interpolate(b_up2, size=out_8x.shape[2:]) + out_8x
        b_up4 = F.interpolate(b_up3, size=out_4x.shape[2:]) + out_4x

        # TOP - BOTTOM
        t_down1 = out_4x
        t_down2 = F.interpolate(t_down1, size=out_8x.shape[2:]) + out_8x
        t_down3 = F.interpolate(t_down2, size=out_16x.shape[2:]) + out_16x
        t_down4 = F.interpolate(t_down3, size=out_32x.shape[2:]) + out_32x

        # P
        P4 = F.leaky_relu(self.P4(b_up4 + t_down1))
        P8 = F.leaky_relu(self.P8(b_up3 + t_down2))
        P16 = F.leaky_relu(self.P16(b_up2 + t_down3))
        P32 = F.leaky_relu(self.P32(b_up1 + t_down4))

        # ----------------Semantic--------------------------------
        semantic_logits = self.semantic_head(P4, P8, P16, P32)

        # -----------------Instance--------------------------
        # images = F.interpolate(img, size=self.original_image_size)
        # feature_maps = OrderedDict([('P4', P4),
        #                             ('P8', P8),
        #                             ('P16', P16),
        #                             ('P32', P32)])

        # image_list = ImageList(images, [x.shape[1:] for x in images])
        # image_sizes = [x.shape[1:] for x in images]

        # proposals, proposal_losses = self.rpn(
        #     image_list, feature_maps, anns)

        # roi_result, roi_losses = self.roi_pool(
        #     feature_maps, proposals, image_sizes, targets=anns)

        # roi_result = self.transform.postprocess(
        #     roi_result, image_sizes, image_sizes)
        # -----------------------------------------------
        out = self.output_layer(fused_5)

        out = out.squeeze_(1)

        if self.training:

            mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=temp_variables.DEVICE,
                                                                    dtype=torch.float64), torch.tensor((0), device=temp_variables.DEVICE, dtype=torch.float64))
            mask_gt = mask_gt.squeeze_(1)
            mask_gt.requires_grad_(True)
            sparse_depth_gt = sparse_depth_gt.squeeze_(
                1)  # remove C dimension there's only one

            loss = F.mse_loss(out*mask_gt, sparse_depth_gt*mask_gt)

            # --------Semantic--------------

            semantic_masks = list(
                map(lambda ann: ann['semantic_mask'], anns))
            semantic_masks = tensorize_batch(
                semantic_masks, temp_variables.DEVICE)

            losses["semantic_loss"] = F.cross_entropy(
                semantic_logits, semantic_masks.long())

            losses = {**losses, "depth_loss": loss}

            # losses = {"depth_loss": loss}

            return losses, out

        else:
            return None, [{'semantic_logits': semantic_logits[idx], 'depth': out[idx]} for idx, _ in enumerate(img)]
            # return None, out
        #
