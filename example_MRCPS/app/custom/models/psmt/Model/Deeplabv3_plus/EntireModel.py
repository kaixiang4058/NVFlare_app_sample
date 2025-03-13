import torch.nn
from models.psmt.Utils.losses import *
from models.psmt.Base.base_model import BaseModel
from models.psmt.Model.Deeplabv3_plus.encoder_decoder import *

res_net = "CityCode/Model/PSPNet/Backbones/pretrained/3x3resnet{}-imagenet.pth"
res_net_2 = "models/psmt/Model/Deeplabv3_plus/Backbones/pretrained/resnet{}.pth"

class EntireModel(BaseModel):
    def __init__(self, num_classes, data_shape=(512,512), sup_loss=torch.nn.CrossEntropyLoss, cons_w_unsup=None,
                 pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):
        super(EntireModel, self).__init__()
        self.encoder1 = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d)
        self.decoder1 = DecoderNetwork(num_classes=num_classes, data_shape=data_shape)
        self.encoder2 = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d)
        self.decoder2 = DecoderNetwork(num_classes=num_classes, data_shape=data_shape)
        self.encoder_s = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                        pretrained_model=res_net_2.format(str(50)))
        self.decoder_s = VATDecoderNetwork(num_classes=num_classes, data_shape=data_shape)
        self.mode = "semi"
        self.sup_loss = sup_loss
        self.unsup_loss_w = cons_w_unsup
        self.unsuper_loss = semi_ce_loss

    def freeze_teachers_parameters(self):
        for p in self.encoder1.parameters():
            p.requires_grad = False
        for p in self.decoder1.parameters():
            p.requires_grad = False

        for p in self.encoder2.parameters():
            p.requires_grad = False
        for p in self.decoder2.parameters():
            p.requires_grad = False

    def warm_up_forward(self, id, x, y):
        if id == 1:
            output_l = self.decoder1(self.encoder1(x))
        elif id == 2:
            output_l = self.decoder2(self.encoder2(x))
        else:
            output_l = self.decoder_s(self.encoder_s(x))

        loss = F.cross_entropy(output_l, y)
        curr_losses = {'loss_sup': loss}
        outputs = {'sup_pred': output_l}
        return loss, curr_losses, outputs

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, id=0,
                warm_up=False, lam=0, pad=None, semi_p_th=0.6, semi_n_th=0.6):
        if warm_up:
            return self.warm_up_forward(id=id, x=x_l, y=target_l)

        output_l = self.decoder_s(self.encoder_s(x_l), t_model=[self.decoder1, self.decoder2])
        # Supervised loss
        loss_sup = F.cross_entropy(output_l, target_l)
        curr_losses = {'loss_sup': loss_sup}
        output_ul = self.decoder_s(self.encoder_s(x_ul), t_model=[self.decoder1, self.decoder2])
        loss_unsup, pass_rate, neg_loss = self.unsuper_loss(inputs=output_ul, targets=target_ul,
                                                            conf_mask=True, threshold=semi_p_th,
                                                            threshold_neg=semi_n_th)

        # for negative learning
        if semi_n_th > .0:
            confident_reg = .5 * torch.mean(F.softmax(output_ul, dim=1) ** 2)
            loss_unsup += neg_loss
            loss_unsup += confident_reg

        loss_unsup = loss_unsup * self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
        total_loss = loss_unsup + loss_sup

        curr_losses['loss_unsup'] = loss_unsup
        curr_losses['pass_rate'] = pass_rate
        curr_losses['neg_loss'] = neg_loss
        outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}
        return total_loss, curr_losses, outputs

