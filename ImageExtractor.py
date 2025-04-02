import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights


def l2norm(X):
    "L2-normalize columns of X"
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class ImgEncoderGlobal(nn.Module):
    def __init__(self, embed_size, finetune=True, use_abs=False, no_imgnorm=False):
        "Load pretrained ResNet18 and replace top fc layer."
        super(ImgEncoderGlobal, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained ResNet18 model
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of ResNet18 with a new one
        self.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        self.cnn.fc = nn.Sequential()

        self.init_weights()

    def init_weights(self):
        "Xavier's initialization for the fully connected layer"
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        "Extract image feature vectors."
        # Extracting global features using ResNet18
        global_features = self.cnn(images)

        # Normalization in the image embedding space
        global_features = l2norm(global_features)

        # Linear projection to the joint embedding space
        global_features = self.fc(global_features)

        # Normalization in the joint embedding space
        if not self.no_imgnorm:
            global_features = l2norm(global_features)

        # Take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            global_features = torch.abs(global_features)

        return global_features


class ImgEncoderLocal(nn.Module):
    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        "Initialize for handling precomputed image features."
        super(ImgEncoderLocal, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        "Xavier's initialization for the fully connected layer"
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        "Extract image feature vectors."
        # Extracting local features from precomputed features
        local_features = self.fc(images)

        # Normalization in the joint embedding space
        if not self.no_imgnorm:
            local_features = l2norm(local_features)

        # Take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            local_features = torch.abs(local_features)

        return local_features
