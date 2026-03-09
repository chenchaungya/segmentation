
from model.DBNet import DBNet


def build_model(model_name, num_classes):
    if model_name == 'DBNet':
        return DBNet(classes=num_classes)

