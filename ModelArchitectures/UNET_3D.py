import torch
import torch.nn as nn

def encoder_block(in_channels, out_channels, dropout_rate=0.1):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
    )

def decoder_block(in_channels, out_channels, dropout_rate=0.1):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels,kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout_rate),
    )

def upconv_block(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)

class UNET_3D(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, init_features = 32, dropout_rate=0.1):
        super(UNET_3D, self).__init__()

        # Descending part of the U : the encoding part
        self.encoder1 = encoder_block(in_channels, init_features, dropout_rate)
        self.encoder2 = encoder_block(init_features, init_features*2, dropout_rate)
        self.encoder3 = encoder_block(init_features*2, init_features*4, dropout_rate)
        self.encoder4 = encoder_block(init_features*4, init_features*8, dropout_rate)

        # The bootleneck part
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=init_features*8, out_channels=init_features*16, kernel_size=3, padding=1),
            nn.BatchNorm3d(init_features*16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            nn.Conv3d(in_channels=init_features*16, out_channels=init_features*16, kernel_size=3, padding=1),
            nn.BatchNorm3d(init_features*16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

        self.upconv1 = upconv_block(init_features*16, init_features*8)
        self.decoder1 = decoder_block(init_features*16, init_features*8, dropout_rate)

        self.upconv2 = upconv_block(init_features*8, init_features*4)
        self.decoder2 = decoder_block(init_features*8, init_features*4, dropout_rate)

        self.upconv3 = upconv_block(init_features*4, init_features*2)
        self.decoder3 = decoder_block(init_features*4, init_features*2, dropout_rate)

        self.upconv4 = upconv_block(init_features*2, init_features)
        self.decoder4 = decoder_block(init_features*2, init_features, dropout_rate)

        self.final_conv = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding part
        e1 = self.encoder1(x)
        p1 = nn.MaxPool3d(kernel_size=2, stride=2)(e1) # different variable because we'll need the other value for the decoding steps

        e2 = self.encoder2(p1)
        p2 = nn.MaxPool3d(kernel_size=2, stride=2)(e2)
        
        e3 = self.encoder3(p2)
        p3 = nn.MaxPool3d(kernel_size=2, stride=2)(e3)

        e4 = self.encoder4(p3)
        p4 = nn.MaxPool3d(kernel_size=2, stride=2)(e4)

        # Bottom of the U : bottleneck
        b = self.bottleneck(p4)

        # Decoding part
        d1 = self.upconv1(b)
        d1 = torch.cat((d1, e4), dim=1) # concatenate the output of the encoder with the input of the incoder
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.decoder2(d2)
        
        d3 = self.upconv3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3(d3)

        d4 = self.upconv4(d3)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.decoder4(d4)

        output = self.final_conv(d4)
        #output = torch.sigmoid(output) # because want a binary output : 0 no aliasing and 1 aliasing

        return output