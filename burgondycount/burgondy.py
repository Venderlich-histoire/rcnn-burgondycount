import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

# setting root
ROOT_DIR = "./"
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

#insert mrcnn lib
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import tensorflow as tf


#Save Log
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#dl coco trained weight from release
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class BurgondyConfig(Config):
    NAME = "burgondy_segment"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 

    NUM_CLASSES = 1 + 3

    IMAGE_MAX_DIM = 515
    IMAGE_MAX_DIM = 512

    STEPS_PER_EPOCH = 1
    VALIDATION_STEPS = 1
    BACKBONE = 'resnet50'
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 1
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 1 
    POST_NMS_ROIS_TRAINING = 1

config = BurgondyConfig()
config.display()

class BurgondyDataSet(utils.Dataset):
    #Load coco annotation
    def load_data(self, annotation_json, images_dir):
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close

        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
            MaskRCNN expects masks in the form of a bitmap [height, width, instances].
            Args:
            image_id: The id of the image to load masks for
            Returns:
            masks: A bool array of shape [height, width, instance count] with
            one mask per instance. class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info=self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks= []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new ('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)
        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids

dataset_train = BurgondyDataSet()
dataset_train.load_data(ROOT_DIR+'samples/burgondycount/datasets/train/coco_annotations.json', ROOT_DIR+'samples/burgondycount/datasets/train')
dataset_train.prepare()

dataset_val = BurgondyDataSet()
dataset_val.load_data(ROOT_DIR+'samples/burgondycount/datasets/val/coco_annotations.json', ROOT_DIR+'samples/burgondycount/datasets/val')
dataset_val.prepare()


dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 4)

for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
 
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "coco"

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)

start_train = time.time()
model.train(dataset_train, 
            dataset_val,
            learning_rate= config.LEARNING_RATE,
            epochs=4,
            layers='heads')
end_train = time.time()
minutes = round ((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')


class InferenceConfig(BurgondyConfig):
    GPU_COUNT = 1
    Images_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

import skimage
real_test_dir = '../../../datasets/cig_butts/real_test/'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], figsize=(5,5))