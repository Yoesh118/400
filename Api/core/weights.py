from .mrcnn.config import Config
from .mrcnn.model import MaskRCNN


# define the test configuration
class PlatesConfig(Config):
    NAME = 'object'

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1+37

    STEPS_PER_EPOCH = 50

    BATCH_SIZE = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class Model:
    def __init__(self):
        self.model = self.loadWeights()

    @staticmethod
    def loadWeights():
        rcnn = MaskRCNN(mode='inference', model_dir='core/load_model/weights.h5', config=PlatesConfig())
        rcnn.load_weights(by_name=True)
        return rcnn
