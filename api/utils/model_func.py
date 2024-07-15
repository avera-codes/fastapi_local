import torch
from torchvision.models import resnet18
import torchvision.transforms as T
import json

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def load_classes():
    '''
    Returns IMAGENET classes
    '''
    with open('utils/imagenet-simple-labels.json') as f:
        labels = json.load(f)
    return labels

def class_id_to_label(i):
    '''
    Input int: class index
    Returns class name
    '''

    labels = load_classes()
    return labels[i]

def load_model():
    '''
    Returns resnet model with IMAGENET weights
    '''
    model = resnet18()
    model.load_state_dict(torch.load('utils/resnet18-weights.pth', map_location='cpu'))
    model.eval()
    return model

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    return trnsfrms(img)


