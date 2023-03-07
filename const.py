from enum import Enum


class AnomalyID(Enum):
    abnormal = 1
    normal = 0


class CardID(Enum):
    CitizenCardV1_back = 0
    CitizenCardV1_front = 1
    CitizenCardV2_back = 2
    CitizenCardV2_front = 3
    IdentificationCard_back = 4
    IdentificationCard_front = 5