import torch
import torch.nn as nn
import torchvision.models as models

class RetinaModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(RetinaModel, self).__init__()
        
        # Use ResNet50 as base model
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'conv1' in name or 'bn1' in name:
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes=5, device='cuda'):
    model = RetinaModel(num_classes=num_classes)
    return model.to(device)