import time
from keras.preprocessing.image import load_img, img_to_array
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import os
import warnings

# from mrcnn.visualize import display_instances
from mrcnn.visualize import display_instances

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
class_names =[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
            "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z"
        ]

class CardConfig(Config):
    NAME = 'object'

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1+37

    STEPS_PER_EPOCH = 50

    BATCH_SIZE = 1


    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


def main():
    ###################################################################################
    # load this on boot
    config = CardConfig()
    rcnn = modellib.MaskRCNN(mode="inference", model_dir="./", config=config)
    rcnn.load_weights('mask_rcnn_object_0049.h5', by_name=True)
    img = img_to_array(load_img("testim.jpg"))
    print(img.shape)
    r = rcnn.detect([img], verbose=1)[0]
    print(r)
    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    print(r)

main()
