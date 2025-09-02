import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="resnet50", num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )
        # self.model = torch.compile(self.model, backend="inductor")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        x = self.model(x)
        return x


class Mixed_Encoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="resnet50", num_classes=339, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool=""
        )
        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Create the classifier
        if hasattr(self.model, "num_features"):
            num_features = self.model.num_features
        else:
            # Fallback, can be adjusted based on the specific model
            num_features = 2048

        self.classifier = nn.Linear(num_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        # Extract features
        features = self.model(x)

        # Pool the features to make them of fixed size
        pooled_features = self.global_pool(features).flatten(1)

        # Classify
        logits = self.classifier(pooled_features)
        # print('logits', logits.shape)
        # print('pooled_features', pooled_features.shape)
        return logits, pooled_features
