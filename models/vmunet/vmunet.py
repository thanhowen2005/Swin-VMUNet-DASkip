from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 decoder_type='cnn',
                 deep_supervision=True,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
            decoder_type=decoder_type,
            deep_supervision=deep_supervision,
        )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = self.vmunet(x)
        if isinstance(out, list):
            if self.num_classes == 1:
                return [torch.sigmoid(o) for o in out]
            else:
                return out
        else:
            if self.num_classes == 1:
                return torch.sigmoid(out)
            else:
                return out
    
    def freeze_encoder(self):
        self.vmunet.freeze_encoder()

    def unfreeze_encoder(self):
        self.vmunet.unfreeze_encoder()

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu', weights_only=False)
            pretrained_dict = modelCheckpoint['model']
            
            # Load encoder pretrained weights
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)
            ))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys count:', len(not_loaded_keys))
            print(">>> [Swin-VMUNet] ImageNet Pretrained Encoder successfully loaded finished!")