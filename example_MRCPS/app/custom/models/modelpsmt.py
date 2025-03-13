import torch

import segmentation_models_pytorch.utils as smputils

from collections import OrderedDict

from .psmt.Model.Deeplabv3_plus.EntireModel import EntireModel as Deeplabv3_plus
from .psmt.Model.SegFormer.EntireModel import EntireModel as SegFormer
from .psmt.Model.UnetRs26d.EntireModel import EntireModel as UnetRs26d

from models.psmt.Utils.losses import consistency_weight
from itertools import chain

from . import transformsgpu
import random 
from .modelbase import BaseModel

class PSMTModel(BaseModel):
    def __init__(self, traincfg):
        super().__init__(traincfg)

        self.unflatten_json(self.traincfg['branch1'])
        model_type = self.traincfg['branch1']['model'].pop('type')
        torch.manual_seed(self.traincfg['branch1']['model_seed'])

        if model_type == "SegFormer":
            self.model = SegFormer(num_classes=2, sup_loss=self.criterion)
        elif model_type == "Unet":
            self.model = UnetRs26d(num_classes=2, sup_loss=self.criterion)
        elif model_type == "DeepLabV3Plus":
            self.model = Deeplabv3_plus(num_classes=2, sup_loss=self.criterion)

        self.gamma = 0.5
        self.num_classes = len(self.traincfg['classes'])

    def setup(self, stage):
        if stage == 'fit':
            super().setup(stage)
            cons_w_unsup = consistency_weight(final_w=self.consistencyratio, iters_per_epoch=self.steps_per_epoch,
                                      rampup_starts=0, rampup_ends=4)
            self.model.unsup_loss_w = cons_w_unsup

    def forward(self, x, step=1):
        # in lightning, forward defines the prediction/inference actions
        """
        Args:
            x       input tensor
            step    predict branch
        """
        encoder = getattr(self.model, f'encoder{step}')
        decoder = getattr(self.model, f'decoder{step}')

        y_pred = decoder(encoder(x),
                    data_shape=[x.shape[-2], x.shape[-1]])

        return torch.argmax(y_pred, dim=1)

    @torch.no_grad()
    def update_teachers(self, teacher_encoder, teacher_decoder, keep_rate=0.996):
        student_encoder_dict = self.model.encoder_s.state_dict()
        student_decoder_dict = self.model.decoder_s.state_dict()
        new_teacher_encoder_dict = OrderedDict()
        new_teacher_decoder_dict = OrderedDict()

        for key, value in teacher_encoder.state_dict().items():

            if key in student_encoder_dict.keys():
                new_teacher_encoder_dict[key] = (
                        student_encoder_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student encoder model".format(key))

        for key, value in teacher_decoder.state_dict().items():

            if key in student_decoder_dict.keys():
                new_teacher_decoder_dict[key] = (
                        student_decoder_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student decoder model".format(key))
        teacher_encoder.load_state_dict(new_teacher_encoder_dict, strict=True)
        teacher_decoder.load_state_dict(new_teacher_decoder_dict, strict=True)

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul1 = self.model.decoder1(self.model.encoder1(image),
                                                            data_shape=[image.shape[-2], image.shape[-1]])
            predict_target_ul2 = self.model.decoder2(self.model.encoder2(image),
                                                            data_shape=[image.shape[-2], image.shape[-1]])
            predict_target_ul1 = torch.nn.functional.interpolate(predict_target_ul1,
                                                                 size=(image.shape[-2], image.shape[-1]),
                                                                 mode='bilinear',
                                                                 align_corners=True)

            predict_target_ul2 = torch.nn.functional.interpolate(predict_target_ul2,
                                                                 size=(image.shape[-2], image.shape[-1]),
                                                                 mode='bilinear',
                                                                 align_corners=True)

            assert predict_target_ul1.shape == predict_target_ul2.shape, "Expect two prediction in same shape,"
        return predict_target_ul1, predict_target_ul2

    # NOTE: the func in here doesn't bring improvements, but stabilize the early stage's training curve.
    def assist_mask_calculate(self, core_predict, assist_predict, topk=1):
        _, index = torch.topk(assist_predict, k=topk, dim=1)
        mask = torch.nn.functional.one_hot(index.squeeze())
        # k!= 1, sum them
        mask = mask.sum(dim=1) if topk > 1 else mask
        if mask.shape[-1] != self.num_classes:
            expand = torch.zeros(
                [mask.shape[0], mask.shape[1], mask.shape[2], self.num_classes - mask.shape[-1]]).cuda()
            mask = torch.cat((mask, expand), dim=3)
        mask = mask.permute(0, 3, 1, 2)
        # get the topk result of the assist map
        assist_predict = torch.mul(assist_predict, mask)

        # fullfill with core predict value for the other entries;
        # as it will be merged based on threshold value
        assist_predict[torch.where(assist_predict == .0)] = core_predict[torch.where(assist_predict == .0)]
        return assist_predict

    def on_train_epoch_start(self):
        self.model.freeze_teachers_parameters()

    def training_step(self, batch, batch_idx):
        # self.model.freeze_teachers_parameters()
        input_l, target_l = batch["label"]
        input_ul_wk = batch["unlabel"]

        predmask = []
        s_opt = self.optimizers()

        # predicted unlabeled data
        t1_prob, t2_prob = self.predict_with_out_grad(input_ul_wk)

        # if self.current_epoch % 2 == 0:
        #     t2_prob = self.assist_mask_calculate(core_predict=t1_prob,
        #                                         assist_predict=t2_prob,
        #                                         topk=1)

        # else:
        #     t1_prob = self.assist_mask_calculate(core_predict=t2_prob,
        #                                         assist_predict=t1_prob,
        #                                         topk=1)
        predict_target_ul = self.gamma * t1_prob + (1 - self.gamma) * t2_prob

        with torch.no_grad():
            pseudomask_un_t = predict_target_ul
            MixMask = self._returnCutMask(pseudomask_un_t.shape[2:4], pseudomask_un_t.shape[0], cut_type="cut")
            strong_parameters = {"Cut": MixMask}
            strong_parameters["flip"] = random.randint(0, 7)
            strong_parameters["ColorJitter"] = random.uniform(0, 1)

            mix_un_img, mix_un_mask = self._strongTransform(
                                                strong_parameters,
                                                data=input_ul_wk,
                                                target=pseudomask_un_t,
                                                cutaug=transformsgpu.mix
                                                )

        totalloss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l,
                                                    x_ul=mix_un_img,
                                                    target_ul=mix_un_mask,
                                                    curr_iter=self.global_step, epoch=self.current_epoch,
                                                    semi_p_th=0.6,
                                                    semi_n_th=0.6)

        predmask.append(torch.argmax(outputs['sup_pred'], dim=1))
        self.log(f"train loss", totalloss, prog_bar=True)
        self.log(f"train sup loss", cur_losses['loss_sup'])
        self.log(f"train cr loss", cur_losses['loss_unsup'])

        self._evaluate(predmask, target_l, "train")
        self.manual_backward(totalloss/self.accumulate_grad_batches)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            s_opt.step()
            s_opt.zero_grad()

            with torch.no_grad():
                if self.global_step // 300 % 2 == 0:
                    self.update_teachers(teacher_encoder=self.model.encoder1,
                                        teacher_decoder=self.model.decoder1)
                else:
                    self.update_teachers(teacher_encoder=self.model.encoder2,
                                        teacher_decoder=self.model.decoder2)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._training_sch_on_step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predmask = []
        y_pred_1 = self.model.decoder1(self.model.encoder1(x),
                                data_shape=[x.shape[-2], x.shape[-1]])
        predmask.append(torch.argmax(y_pred_1, dim=1))

        loss_1 = self.criterion(y_pred_1, y)
        self.log(f"valid loss", loss_1, prog_bar=True)

        self._evaluate(predmask, y, "valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        predmask = []
        y_pred_1 = self.model.decoder1(self.model.encoder1(x),
                                data_shape=[x.shape[-2], x.shape[-1]])
        predmask.append(torch.argmax(y_pred_1, dim=1))
        
        self._evaluate(predmask, y, "test")

    def configure_optimizers(self):
        opts = []
        schs = []

        optimizer1 = self._initoptimizer(chain(self.model.encoder_s.parameters(), self.model.decoder_s.parameters()))
        scheduler1 = self._initscheduler(optimizer1)
        opts.append(optimizer1)
        schs.append(scheduler1)

        return opts, schs

    def _initmetrics(self):
        return [
            smputils.metrics.IoU(),
        ]