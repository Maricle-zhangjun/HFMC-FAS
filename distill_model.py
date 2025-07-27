import torch
import torch.nn as nn
import torch.nn.functional as F
from student_model import ViT_Tiny_Patch16


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # LGFE Module
        # First branch (ResNet-like)-->LFE Module
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Second branch (ViT-like)-->GFE Module
        self.multihead_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=12)
        self.norm1 = nn.LayerNorm([14, 14, out_channels])
        self.feedforward = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm2 = nn.LayerNorm([14, 14, out_channels])

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # First branch
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = out1 + self.shortcut(identity)
        out1 = self.relu1(out1)
        
        # Second branch
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, 14 * 14, self.multihead_attention.embed_dim)  # Reshape for Multihead Attention
        out2, _ = self.multihead_attention(x_reshaped, x_reshaped, x_reshaped)
        out2 = out2.view(x.size())  # Reshape back to (batch_size, channels, height, width)
        out2 = self.norm1(out2.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        out2_middle = out2
        out2 = out2 + self.shortcut(identity)
        
        out2 = self.feedforward(out2.view(-1, self.multihead_attention.embed_dim)).view(x.size())
        out2 = self.norm2((out2 + x).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        out2 = out2 + self.shortcut(out2_middle)
        out2 = out2 + self.shortcut(identity)
        
        out = 0.5*out1 + 0.5*out2
        return out

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # FIFE Module
        #  Mutil_Conv_5×5 + Conv_1×1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0)

        # Feature extraction layers
        self.feature_layers = nn.ModuleList([
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
            BasicBlock(768, 768),
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Feature extraction layers
        features = []
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            features.append(x)

        return x, features

class teacher_model(nn.Module):
    def __init__(self):
        super(teacher_model, self).__init__()

        # Feature extraction
        self.feature_extractor = FeatureExtractor()

        # Classification layer
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x, features = self.feature_extractor(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x, features

class DistillationModel(nn.Module):
    def __init__(self):
        super(DistillationModel, self).__init__()
        self.teacher = teacher_model()
        self.student = ViT_Tiny_Patch16()
        

    def forward(self, x):
        teacher_output, teacher_features = self.teacher(x)
        student_output, student_features = self.student(x)

        return teacher_output,teacher_features,student_output,student_features
    
def TSmodel():
    return DistillationModel()

def test_model():
    model = TSmodel()
    model.eval()  # Set the model to evaluation mode
    input_tensor = torch.randn(2, 3, 224, 224)  # Initialize a random tensor
    with torch.no_grad():  # Disable gradient calculation for inference
        teacher_output, teacher_features, student_output, student_features = model(input_tensor)
    print(teacher_features[0].shape) # torch.Size([2, 768, 14, 14])
    print(student_features[0].shape) # torch.Size([2, 196, 192])
    teacher_features[0]=teacher_features[0].reshape(2, 768, 196)
    teacher_features[0]=torch.transpose(teacher_features[0],1,2)
    print(teacher_features[0].shape) # torch.Size([2, 196, 768])
    print(student_features[0].shape)
    print(teacher_output.shape) # torch.Size([2, 2])
    print(student_output.shape) # torch.Size([2, 2])

if __name__ == "__main__":
    test_model()