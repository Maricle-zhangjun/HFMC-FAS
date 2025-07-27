import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_Tiny_Patch16(nn.Module):
    def __init__(self, num_classes=2, dim=192, depth=12, heads=3, mlp_dim=768, dropout=0.1):
        super(ViT_Tiny_Patch16, self).__init__()
        
        # Patch Embedding
        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Class Token
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embedding(x)  # [batch_size, dim, num_patches_h, num_patches_w]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, dim]
        
        # Add Class Token
        class_token = self.class_token.expand(batch_size, -1, -1)  # [batch_size, 1, dim]
        x = torch.cat([class_token, x], dim=1)  # [batch_size, num_patches + 1, dim]
        
        # Add Positional Embedding
        x = x + self.positional_embedding
        
        # Store outputs from each encoder layer
        encoder_outputs = []
        
        # Transformer Encoder
        for layer in self.transformer_encoder.layers:
            x = layer(x)
            encoder_outputs.append(x[:, 1:, :])  # Save each layer's output
        
        # Extract Class Token
        x = x[:, 0]  # [batch_size, dim]
        
        # MLP Head
        x = self.mlp_head(x)  # [batch_size, num_classes]
        
        return x, encoder_outputs  # Return both the final output and the encoder layer outputs

# Example usage
if __name__ == "__main__":
    model = ViT_Tiny_Patch16(num_classes=2)
    input_tensor = torch.randn(8, 3, 224, 224)  # [batch_size, channels, height, width]
    output, encoder_features = model(input_tensor)
    print(output.shape) # torch.Size([8, 2])
    print(len(encoder_features)) # 12
    print(encoder_features[0].shape) # torch.Size([8, 196, 192]) 
