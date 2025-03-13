import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch.utils as smputils
import math
import numpy as np
import scipy.ndimage
import copy


from .modelbase import BaseModel

class GCTModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        self.traincfg['branch2'] = copy.deepcopy(self.traincfg['branch1'])
        self.traincfg['branch2']['model_seed'] *= 2

        self.l_model = self._initmodel(self.traincfg['branch1'])
        self.r_model = self._initmodel(self.traincfg['branch2'])

        self.fd_model = FlawDetector(self.traincfg['branch1']['model']['classes']+3)

        self.fd_criterion = FlawDetectorCriterion()
        self.dc_criterion = torch.nn.MSELoss()

        self.zero_df_gt = torch.zeros([self.traincfg['traindl']['batchsize'], 1, \
            self.traincfg['traindl']['patchsize'], self.traincfg['traindl']['patchsize']])
        # build the extra modules required by GCT
        self.flawmap_handler = FlawmapHandler(self.traincfg['traindl']['patchsize'])
        self.dcgt_generator = DCGTGenerator()
        self.fdgt_generator = FDGTGenerator(self.traincfg['traindl']['patchsize'])

    def training_step(self, batch, batch_idx):
        x, y = batch["label"]
        x_un = batch["unlabel"]

        lbs = len(x)
        x = torch.concat((x, x_un), dim=0)

        predmask = []
        l_opt, r_opt, fd_opt = self.optimizers()


        # calculate the ramp-up coefficient of the dynamic consistency constraint
        dc_rampup_epochs = 1
        cur_steps = self.trainer.global_step
        total_steps = self.steps_per_epoch * dc_rampup_epochs
        dc_rampup_scale = self.sigmoid_rampup(cur_steps, total_steps)
        # supervised
        # -----------------------------------------------------------------------------
        # step-0: pre-forwarding to save GPU memory
        #   - forward the task models and the flaw detector
        #   - generate pseudo ground truth for the unlabeled data if the dynamic
        #     consistency constraint is enabled
        # -----------------------------------------------------------------------------
        with torch.no_grad():
            l_activated_pred = F.softmax(self.l_model(x), dim=1)
            r_activated_pred = F.softmax(self.r_model(x), dim=1)
        
        # evaluation
        predmask.append(torch.argmax(l_activated_pred[:lbs], dim=1))
        predmask.append(torch.argmax(r_activated_pred[:lbs], dim=1))
        self._evaluate(predmask, y, "train")

        # 'l_flawmap' and 'r_flawmap' will be used in step-2
        l_flawmap = self.fd_model(x, l_activated_pred)
        r_flawmap = self.fd_model(x, r_activated_pred)
        
        l_dc_gt, r_dc_gt = None, None
        l_fc_mask, r_fc_mask = None, None
        
        # generate the pseudo ground truth for the dynamic consistency constraint
        with torch.no_grad():
            l_handled_flawmap = self.flawmap_handler(l_flawmap)
            r_handled_flawmap = self.flawmap_handler(r_flawmap)
            l_dc_gt, r_dc_gt, l_fc_mask, r_fc_mask = self.dcgt_generator(
                l_activated_pred.detach(), r_activated_pred.detach(), l_handled_flawmap, r_handled_flawmap)

        # -----------------------------------------------------------------------------
        # step-1: train the task models
        # -----------------------------------------------------------------------------
        for param in self.fd_model.parameters():
            param.requires_grad = False

        # train the 'l' task model
        l_loss = self._task_model_iter('l', lbs, x, y, l_dc_gt, l_fc_mask, dc_rampup_scale)
        self.log(f"train 1 loss", l_loss)

        # train the 'r' task model
        r_loss = self._task_model_iter('r', lbs, x, y, r_dc_gt, r_fc_mask, dc_rampup_scale)
        self.log(f"train 2 loss", r_loss)

        totalloss = l_loss + r_loss
        self.log(f"train loss", totalloss)

        self.manual_backward(totalloss / self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            l_opt.step()
            l_opt.zero_grad()
            r_opt.step()
            r_opt.zero_grad()

        # -----------------------------------------------------------------------------
        # step-2: train the flaw detector
        # -----------------------------------------------------------------------------
        for param in self.fd_model.parameters():
            param.requires_grad = True
        
        # generate the ground truth for the flaw detector (on labeled data only)
        with torch.no_grad():
            mask = torch.unsqueeze(y, dim=1)
            l_flawmap_gt = self.fdgt_generator.forward(
                l_activated_pred[:lbs, ...].detach(), torch.cat((1-mask,mask), dim=1))
            r_flawmap_gt = self.fdgt_generator.forward(
                r_activated_pred[:lbs, ...].detach(), torch.cat((1-mask,mask), dim=1)) 
        
        l_fd_loss = self.fd_criterion.forward(l_flawmap[:lbs, ...], l_flawmap_gt)
        l_fd_loss = torch.mean(l_fd_loss)
        
        r_fd_loss = self.fd_criterion.forward(r_flawmap[:lbs, ...], r_flawmap_gt)
        r_fd_loss = torch.mean(r_fd_loss)

        fd_loss = (l_fd_loss + r_fd_loss) / 2

        self.manual_backward(fd_loss / self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            fd_opt.step()
            fd_opt.zero_grad()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def _task_model_iter(self, mid, lbs, 
                         inp, gt, dc_gt, fc_mask, dc_rampup_scale):

        model = getattr(self, f"{mid}_model")

        # forward the task model
        pred = model(inp)
        
        activated_pred = F.softmax(pred, dim=1)

        flawmap = self.fd_model(inp, activated_pred)

        # calculate the supervised task constraint on the labeled data
        labeled_pred = pred[:lbs]
        labeled_gt = gt[:lbs]
        task_loss = self.criterion(labeled_pred, labeled_gt)

        # calculate the flaw correction constraint
        if flawmap.shape == self.zero_df_gt.shape:
            fc_ssl_loss = self.fd_criterion(flawmap, self.zero_df_gt.to(self.device), is_ssl=True, reduction=False)
        else:
            fc_ssl_loss = self.fd_criterion(flawmap, torch.zeros(flawmap.shape, device=self.device), is_ssl=True, reduction=False)

        fc_ssl_loss = fc_mask * fc_ssl_loss
        fc_ssl_loss = torch.mean(fc_ssl_loss)

        # calculate the dynamic consistency constraint
        dc_ssl_loss = dc_rampup_scale * self.consistencyratio * self.dc_criterion.forward(activated_pred, dc_gt)

        loss = task_loss + fc_ssl_loss + dc_ssl_loss
        return loss

    def validation_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.l_model(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.r_model(x)
        predmask.append(torch.argmax(y_pred_2, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"valid 1 loss", loss_1)
        loss_2 = self.criterion(y_pred_2, y)
        self.log(f"valid 2 loss", loss_2)

        self.log(f"valid loss", (loss_1 + loss_2) / 2, prog_bar=True)

        self._evaluate(predmask, y, "valid")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, y, "valid ens")

    def test_step(self, batch, batch_idx):
        # return
        x, y = batch
        predmask = []
        y_pred_1 = self.l_model(x)
        predmask.append(torch.argmax(y_pred_1, dim=1))
        y_pred_2 = self.r_model(x)
        predmask.append(torch.argmax(y_pred_2, dim=1))
        
        self._evaluate(predmask, y, "test")

        predensem = []
        voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
        predensem.append(torch.argmax(voting, dim=1))

        self._evaluate(predensem, y, "test ens")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(self.l_model.parameters())
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        optimizer2 = self._initoptimizer(self.r_model.parameters())
        scheduler2 = self._initscheduler(optimizer2)

        opts.append(optimizer2)
        schs.append(scheduler2)

        fd_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.fd_model.parameters()), 
                                       lr=0.0001, betas=(0.9, 0.99))
        schedulerfd = self._initscheduler(fd_optimizer)

        opts.append(fd_optimizer)
        schs.append(schedulerfd)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]

class FlawDetector(nn.Module):
    """ The FC Discriminator proposed in paper:
        'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning'
    """

    ndf = 64    # basic number of channels
	
    def __init__(self, in_channels):
        super(FlawDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, self.ndf, kernel_size=4, stride=2, padding=1)
        self.ibn1 = IBNorm(self.ndf)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1)
        self.ibn2 = IBNorm(self.ndf * 2)
        self.conv2_1 = nn.Conv2d(self.ndf * 2, self.ndf * 2, kernel_size=4, stride=1, padding=1)
        self.ibn2_1 = IBNorm(self.ndf * 2)
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1)
        self.ibn3 = IBNorm(self.ndf * 4)
        self.conv3_1 = nn.Conv2d(self.ndf * 4, self.ndf * 4, kernel_size=4, stride=1, padding=1)
        self.ibn3_1 = IBNorm(self.ndf * 4)
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1)
        self.ibn4 = IBNorm(self.ndf * 8)
        self.conv4_1 = nn.Conv2d(self.ndf * 8, self.ndf * 8, kernel_size=4, stride=1, padding=1)
        self.ibn4_1 = IBNorm(self.ndf * 8)
        self.classifier = nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, task_inp, task_pred):
        # task_inp = torch.cat(task_inp, dim=1)
        x = torch.cat((task_inp, task_pred), dim=1)
        x = self.leaky_relu(self.ibn1(self.conv1(x)))
        x = self.leaky_relu(self.ibn2(self.conv2(x)))
        x = self.leaky_relu(self.ibn2_1(self.conv2_1(x)))
        x = self.leaky_relu(self.ibn3(self.conv3(x)))
        x = self.leaky_relu(self.ibn3_1(self.conv3_1(x)))
        x = self.leaky_relu(self.ibn4(self.conv4(x)))
        x = self.leaky_relu(self.ibn4_1(self.conv4_1(x)))
        x = self.classifier(x)
        x = F.interpolate(x, size=(task_pred.shape[2], task_pred.shape[3]), mode='bilinear', align_corners=True)

        # x is not activated here since it will be activated by the criterion function
        assert x.shape[2:] == task_pred.shape[2:]
        return x


class IBNorm(nn.Module):
    """ This layer combines BatchNorm and InstanceNorm.
    """

    def __init__(self, num_features, split=0.5):
        super(IBNorm, self).__init__()

        self.num_features = num_features
        self.num_BN = int(num_features * split + 0.5)
        self.bnorm = nn.BatchNorm2d(num_features=self.num_BN, affine=True)
        self.inorm = nn.InstanceNorm2d(num_features=num_features - self.num_BN, affine=False)

    def forward(self, x):
        if self.num_BN == self.num_features:
            return self.bnorm(x.contiguous())
        else:
            xb = self.bnorm(x[:, 0:self.num_BN, :, :].contiguous())
            xi = self.inorm(x[:, self.num_BN:, :, :].contiguous())

            return torch.cat((xb, xi), 1)


class FlawDetectorCriterion(nn.Module):
    """ Criterion of the flaw detector.
    """

    def __init__(self):
        super(FlawDetectorCriterion, self).__init__()

    def forward(self, pred, gt, is_ssl=False, reduction=True):    
        loss = F.mse_loss(pred, gt, reduction='none')
        if reduction:
            loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


class FlawmapHandler(nn.Module):
    """ Post-processing of the predicted flawmap.

    This module processes the predicted flawmap to fix some special 
    cases that may cause errors in the subsequent steps of generating
    pseudo ground truth.
    """
    
    def __init__(self, patchsize):
        super(FlawmapHandler, self).__init__()
        self.clip_threshold = 0.1

        blur_ksize = int(patchsize // 16)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

    def forward(self, flawmap):
        flawmap = flawmap.data

        # force all values to be larger than 0
        flawmap.mul_((flawmap >= 0).float())
        # smooth the flawmap
        flawmap = self.blur(flawmap)
        # if all values in the flawmap are less than 'clip_threshold'
        # set the entire flawmap to 0, i.e., no flaw pixel
        fmax = flawmap.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        fmin = flawmap.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_matrix = fmax.repeat(1, 1, flawmap.shape[2], flawmap.shape[3])
        flawmap.mul_((max_matrix > self.clip_threshold).float())
        # normalize the flawmap
        flawmap = flawmap.sub_(fmin).div_(fmax - fmin + 1e-9)

        return flawmap


class DCGTGenerator(nn.Module):
    """ Generate the ground truth of the dynamic consistency constraint.
    """

    def __init__(self):
        super(DCGTGenerator, self).__init__()

    def forward(self, l_pred, r_pred, l_handled_flawmap, r_handled_flawmap):
        l_tmp = l_handled_flawmap.clone()
        r_tmp = r_handled_flawmap.clone()

        l_bad = l_tmp > 0.5
        r_bad = r_tmp > 0.5

        both_bad = (l_bad & r_bad).float()

        l_handled_flawmap.mul_((l_tmp <= 0.5).float())
        r_handled_flawmap.mul_((r_tmp <= 0.5).float())

        l_handled_flawmap.add_((l_tmp > 0.5).float())
        r_handled_flawmap.add_((r_tmp > 0.5).float())

        l_mask = (r_handled_flawmap >= l_handled_flawmap).float()
        r_mask = (l_handled_flawmap >= r_handled_flawmap).float()

        l_dc_gt = l_mask * l_pred + (1 - l_mask) * r_pred
        r_dc_gt = r_mask * r_pred + (1 - r_mask) * l_pred

        return l_dc_gt, r_dc_gt, both_bad, both_bad

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensor

    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor

        Returns:
            torch.Tensor: Blurred version of the input 
        """
            
        return self.op(x)
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

class FDGTGenerator(nn.Module):
    """ Generate the ground truth of the flaw detector, 
        i.e., pipeline 'C' in the paper.
    """

    def __init__(self, patchsize):
        super(FDGTGenerator, self).__init__()

        blur_ksize = int(patchsize / 8)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

        reblur_ksize = int(patchsize / 4)
        reblur_ksize = reblur_ksize + 1 if reblur_ksize % 2 == 0 else reblur_ksize
        self.reblur = GaussianBlurLayer(1, reblur_ksize)

        self.dilate = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        )

    def forward(self, pred, gt):
        diff = torch.abs_(gt - pred.detach())
        diff = torch.sum(diff, dim=1, keepdim=True).mul_(1)
        
        diff = self.blur(diff)
        for _ in range(0, 1):
            diff = self.reblur(self.dilate(diff))

        # normlize each sample to [0, 1]
        dmax = diff.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        dmin = diff.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        diff.sub_(dmin).div_(dmax - dmin + 1e-9)

        flawmap_gt = diff
        return flawmap_gt