from enum import Enum
from torchvision import transforms as T

IMG_SIZE = (300, 450)
# IMG_SIZE = (256, 384)
# IMG_SIZE = (224, 224)

image_transform = T.Compose([T.Resize(IMG_SIZE),
                        #   T.CenterCrop(IMG_SIZE),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])

mask_transform = T.Compose([T.Resize(IMG_SIZE),
                        #  T.CenterCrop(IMG_SIZE),
                            T.ToTensor()
                            ])

class AnomalyID(Enum):
    anomaly = 0
    normal = 1

class QualityID(Enum):
    blur = 0
    occluded = 1
    overexposed = 2
    sharp = 3
    shadow = 4
    outofdate = 5
    lowquality = 6
    overclosed = 7

class CardID(Enum):
    CitizenCardV1_back = 0
    CitizenCardV1_front = 1
    CitizenCardV2_back = 2
    CitizenCardV2_front = 3
    IdentificationCard_back = 4
    IdentificationCard_front = 5
    
nameAno = {1: 'anomaly', 
           0: 'normal'}
