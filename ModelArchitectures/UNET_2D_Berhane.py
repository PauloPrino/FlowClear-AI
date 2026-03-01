import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate):
        super().__init__()
        # Paper Order: Conv -> Dropout -> BN -> ReLU
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out)
        out = self.relu(self.bn(out))
        return torch.cat([x, out], 1) # Concatenate input with output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, nb_layers, growth_rate=12, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(nb_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate, dropout_rate))
            current_channels += growth_rate # The output grows by 'growth_rate' each layer
            
        self.out_channels = current_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class UNET_2D_Berhane(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, growth_rate=12, dropout_rate=0.1):
        super(UNET_2D_Berhane, self).__init__()
        
        # Scaling strategy from the paper: Increase DEPTH (layers), not WIDTH (filters)
        # channels=[2, 4, 6, 8, 10, 12]
        
        # --- ENCODER ---
        # Block 1 (2 layers)
        self.encoder1 = DenseBlock(in_channels, nb_layers=2, growth_rate=growth_rate, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2 (4 layers)
        self.encoder2 = DenseBlock(self.encoder1.out_channels, nb_layers=4, growth_rate=growth_rate, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 (6 layers)
        self.encoder3 = DenseBlock(self.encoder2.out_channels, nb_layers=6, growth_rate=growth_rate, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 (8 layers)
        self.encoder4 = DenseBlock(self.encoder3.out_channels, nb_layers=8, growth_rate=growth_rate, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5 (10 layers)
        self.encoder5 = DenseBlock(self.encoder4.out_channels, nb_layers=10, growth_rate=growth_rate, dropout_rate=dropout_rate)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK (12 layers) ---
        # Paper calls this "conv6" with channels=12
        self.bottleneck = DenseBlock(self.encoder5.out_channels, nb_layers=12, growth_rate=growth_rate, dropout_rate=dropout_rate)

        # --- DECODER ---
        # Note: Paper uses the same dense block logic in decoder
        
        # Up 1
        self.upconv1 = nn.ConvTranspose2d(self.bottleneck.out_channels, 128, kernel_size=2, stride=2) 
        # Skip connection comes from encoder5
        concat_channels = 128 + self.encoder5.out_channels
        self.decoder1 = DenseBlock(concat_channels, nb_layers=10, growth_rate=growth_rate, dropout_rate=dropout_rate)

        # Up 2
        self.upconv2 = nn.ConvTranspose2d(self.decoder1.out_channels, 96, kernel_size=2, stride=2)
        concat_channels = 96 + self.encoder4.out_channels
        self.decoder2 = DenseBlock(concat_channels, nb_layers=8, growth_rate=growth_rate, dropout_rate=dropout_rate)

        # Up 3
        self.upconv3 = nn.ConvTranspose2d(self.decoder2.out_channels, 64, kernel_size=2, stride=2)
        concat_channels = 64 + self.encoder3.out_channels
        self.decoder3 = DenseBlock(concat_channels, nb_layers=6, growth_rate=growth_rate, dropout_rate=dropout_rate)

        # Up 4
        self.upconv4 = nn.ConvTranspose2d(self.decoder3.out_channels, 48, kernel_size=2, stride=2)
        concat_channels = 48 + self.encoder2.out_channels
        self.decoder4 = DenseBlock(concat_channels, nb_layers=4, growth_rate=growth_rate, dropout_rate=dropout_rate)
        
        # Up 5 (To match their 5-layer depth)
        self.upconv5 = nn.ConvTranspose2d(self.decoder4.out_channels, 32, kernel_size=2, stride=2)
        concat_channels = 32 + self.encoder1.out_channels
        self.decoder5 = DenseBlock(concat_channels, nb_layers=2, growth_rate=growth_rate, dropout_rate=dropout_rate)

        # Final Conv
        self.final_conv = nn.Conv2d(self.decoder5.out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        e5 = self.encoder5(p4)
        p5 = self.pool5(e5)
        
        # Bottleneck
        b = self.bottleneck(p5)
        
        # Decoder
        d1 = self.upconv1(b)
        # Center crop if sizes don't match exactly due to pooling
        if d1.shape != e5.shape:
            d1 = torch.nn.functional.interpolate(d1, size=e5.shape[2:])
        d1 = torch.cat([d1, e5], dim=1)
        d1 = self.decoder1(d1)
        
        d2 = self.upconv2(d1)
        if d2.shape != e4.shape:
            d2 = torch.nn.functional.interpolate(d2, size=e4.shape[2:])
        d2 = torch.cat([d2, e4], dim=1)
        d2 = self.decoder2(d2)
        
        d3 = self.upconv3(d2)
        if d3.shape != e3.shape:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d4 = self.upconv4(d3)
        if d4.shape != e2.shape:
            d4 = torch.nn.functional.interpolate(d4, size=e2.shape[2:])
        d4 = torch.cat([d4, e2], dim=1)
        d4 = self.decoder4(d4)
        
        d5 = self.upconv5(d4)
        if d5.shape != e1.shape:
            d5 = torch.nn.functional.interpolate(d5, size=e1.shape[2:])
        d5 = torch.cat([d5, e1], dim=1)
        d5 = self.decoder5(d5)
        
        return self.final_conv(d5)