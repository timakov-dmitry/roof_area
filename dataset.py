import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image
from skimage import draw, io

ROOT_DIR = os.path.abspath("./")
SEASON = 'summer'
COUNT_EPOCHS = 40
TEST_COUNT = 0.85

sys.path.append('./libs')  # 
from libs.config import Config
from libs.mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class RoofConfig(Config):
    NAME = f"roof-{SEASON}-{COUNT_EPOCHS}-"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

class RoofDataset(utils.Dataset):
    
    TRAIN_LIMIT = 0.85

    def load_roof_from_json(self, subset, base_dir):
        with open('./train_markup.json'.format(base_dir)) as json_file:
            cities = json.load(json_file)['cities']
            for next_city in cities:
                city = next_city['title']
                tiles_path = next_city['tiles']
                extension = next_city['extension']
                markup_file = next_city['markup']                
                with open(f'./marcup/{markup_file}') as json_file:
                    tiles = json.load(json_file)
                    tile_keys = list(tiles.keys())
                    train_limit = int(self.TRAIN_LIMIT * len(tile_keys))
                    if subset == 'train':
                        tile_keys = tile_keys[:train_limit]
                    else:
                        tile_keys = tile_keys[train_limit:]   
                    for index, tile_key in enumerate(tile_keys):
                        tile = tiles[tile_key]
                        polygons = tile['regions']
                        if len(polygons) != 0:
                            points = [p['shape_attributes'] for p in polygons if 'all_points_x' in p['shape_attributes']]
                            if len(points) > 0:                        
                                p_str = set([json.dumps(d) for d in points])
                                points = [json.loads(d) for d in p_str]
                                self.add_image(
                                    "roof",
                                    image_id=index, 
                                    path=f"{base_dir}/{city}/tiles/{tile['filename'].split('.')[0]}.{extension}",
                                    width=2048, 
                                    height=2048,
                                    polygons=points)
                    print(f'{city} добавлен ({len(tile_keys)}) тайла.')

    def load_mask(self, image_id):   
        image_info = self.image_info[image_id]    
        if image_info["source"] != "roof":
            return super(self.__class__, self).load_mask(image_id)   
        info = self.image_info[image_id]      
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)        
        for i, p in enumerate(info["polygons"]):  
            rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "roof":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
        

def train(model, base_dir):
    # Training dataset.
    dataset_train = RoofDataset()
    dataset_train.add_class("roof", 1, "roof")
    dataset_train.load_roof_from_json('train', base_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RoofDataset()
    dataset_val.add_class("roof", 1, "roof")
    dataset_val.load_roof_from_json('val', base_dir)    
    dataset_val.prepare()

    print(f'В обучающем датасете {len(dataset_train.image_info)} тайлов')
    print(f'В тестовом датасете {len(dataset_val.image_info)} тайлов')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    return model

