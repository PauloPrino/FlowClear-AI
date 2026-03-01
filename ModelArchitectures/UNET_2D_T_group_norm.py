import torch
import torch.nn as nn
import torch.nn.functional as F

def encoder_block(in_channels, out_channels, dropout_rate=0.1):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(5, out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(5, out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
    )

def decoder_block(in_channels, out_channels, dropout_rate=0.1):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(5, out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(5, out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
    )

def upconv_block(in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

class UNET_2D_T_group_norm(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, dropout_rate=0.1):
        super(UNET_2D_T_group_norm, self).__init__()

        # --- ENCODER ---
        # 3D Ops are used here where Depth = Time
        self.encoder1 = encoder_block(in_channels, init_features, dropout_rate)
        self.encoder2 = encoder_block(init_features, init_features*2, dropout_rate)
        self.encoder3 = encoder_block(init_features*2, init_features*4, dropout_rate)
        self.encoder4 = encoder_block(init_features*4, init_features*8, dropout_rate)

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=init_features*8, out_channels=init_features*16, kernel_size=3, padding=1),
            nn.GroupNorm(5, init_features*16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            nn.Conv3d(in_channels=init_features*16, out_channels=init_features*16, kernel_size=3, padding=1),
            nn.GroupNorm(5, init_features*16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

        # --- DECODER ---
        self.upconv1 = upconv_block(init_features*16, init_features*8, kernel_size=(2,2,2), stride=(2,2,2))
        self.decoder1 = decoder_block(init_features*16, init_features*8, dropout_rate)

        self.upconv2 = upconv_block(init_features*8, init_features*4, kernel_size=(2,2,2), stride=(2,2,2))
        self.decoder2 = decoder_block(init_features*8, init_features*4, dropout_rate)

        self.upconv3 = upconv_block(init_features*4, init_features*2, kernel_size=(1,2,2), stride=(1,2,2))
        self.decoder3 = decoder_block(init_features*4, init_features*2, dropout_rate)

        self.upconv4 = upconv_block(init_features*2, init_features, kernel_size=(1,2,2), stride=(1,2,2))
        self.decoder4 = decoder_block(init_features*2, init_features, dropout_rate)

        self.final_conv = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def _match_size(self, upsampled, bypass):
        """
        Fixes size mismatches caused by odd dimensions (e.g., Time=50).
        If upsampled tensor is smaller than bypass tensor (due to flooring in pooling),
        we interpolate (resize) the upsampled tensor to match.
        """
        if upsampled.shape[2:] != bypass.shape[2:]:
            upsampled = F.interpolate(
                upsampled, 
                size=bypass.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        return upsampled

    def forward(self, x):
        # Encoding part
        e1 = self.encoder1(x)
        # Pooling 1: Keep Time (50), Shrink Space (128)
        p1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))(e1) 

        e2 = self.encoder2(p1)
        # Pooling 2: Keep Time (50), Shrink Space (64)
        p2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))(e2)
        
        e3 = self.encoder3(p2)
        # Pooling 3: Maybe shrink time now? (50 -> 25), Shrink Space (32)
        p3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))(e3)

        e4 = self.encoder4(p3)
        # Pooling 4: Shrink time again (25 -> 12), Shrink Space (16)
        p4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))(e4)

        # Bottleneck Time dimension is now ~12 instead of 3. 
        # Ideally, keep it even higher (e.g. 25 or 50) if GPU memory allows.

        # --- BOTTLENECK ---
        b = self.bottleneck(p4)                             # Out: T=3

        # --- DECODING ---
        # Block 1
        d1 = self.upconv1(b)                                # Out: T=6 (3*2)
        d1 = self._match_size(d1, e4)                       # Safety check (matches 6)
        d1 = torch.cat((d1, e4), dim=1) 
        d1 = self.decoder1(d1)

        # Block 2
        d2 = self.upconv2(d1)                               # Out: T=12 (6*2)
        d2 = self._match_size(d2, e3)                       # Safety check (matches 12)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.decoder2(d2)
        
        # Block 3
        d3 = self.upconv3(d2)                               # Out: T=24 (12*2)
        # CRITICAL FIX: e2 is T=25. upconv output is T=24.
        # _match_size will interpolate d3 from 24 to 25 to allow concatenation.
        d3 = self._match_size(d3, e2)        
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3(d3)

        # Block 4
        d4 = self.upconv4(d3)                               # Out: T=50 (25*2)
        d4 = self._match_size(d4, e1)                       # Matches e1 (T=50)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.decoder4(d4)

        output = self.final_conv(d4)
        
        return output