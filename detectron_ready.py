import os
# fiftyone imports 
import fiftyone as fo
import fiftyone.utils.random as four
import torch
# Detectron imports
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL

import fiftyone.utils.random as four


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

class Detectron2Learner(Learner):
    supported_backbones = ["resnet"]

    def __init__(self, lr=0.00025, batch_size=200, img_per_step=2, weight_decay=0.00008,
                       momentum=0.98, gamma=0.0005, norm="GN", num_workers=2, num_keypoints=25, 
                       iters=4000, threshold=0.9, loss_weight=1.0, device='cuda', temp_path="temp", backbone='resnet'):
        super(Detectron2Learner, self).__init__(lr=lr, threshold=threshold, 
                                                batch_size=batch_size, device=device, 
                                                iters=iters, temp_path=temp_path, 
                                                backbone=backbone)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
        self.cfg.MODEL.MASK_ON = True
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = img_per_step
        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.WEIGHT_DECAY = weight_decay
        self.cfg.SOLVER.GAMMA = gamma
        self.cfg.SOLVER.MOMENTUM = momentum
        self.cfg.SOLVER.MAX_ITER = iters
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size   # faster, and good enough for this toy dataset
        self.cfg.MODEL.SEM_SEG_HEAD.NORM = norm
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = loss_weight

    def __export_split(self, dataset, tag):
        root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/"
        self.cfg.OUTPUT_DIR = root
        extension = ".json"
        samples = dataset.match_tags(tag)
        samples.export(
            dataset_type=fo.types.COCODetectionDataset,
            labels_path= f"{root}{tag}{extension}",
            abs_paths=True,
        )

    def __prepare_dataset(self):
        #four.random_split(dataset, {"train": 0.7, "test": 0.2, "val": 0.1})
        #self.__export_split(dataset, "train")
        #self.__export_split(dataset, "test")
        #self.__export_split(dataset, "val")
        root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/"
        register_coco_instances("diesel_engine_train", {}, root+"train.json", root+"AugImages")
        self.cfg.DATASETS.TRAIN = ("diesel_engine_train",)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.DATASETS.TEST=()    

    def fit(self, verbose=True):
        self.__prepare_dataset()
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        torch.save(self.trainer.model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth"))
        # checkpointer = DetectionCheckpointer(self.trainer.model, save_dir=self.cfg.OUTPUT_DIR)
        # checkpointer.save("model_final.pth")

    def infer(self, img_data):
        pass

    def load(self, verbose=True):
       pass


    def save(self, path, verbose=False):
        pass

    def download(self, path=None, mode="pretrained", verbose=False, 
                        url=OPENDR_SERVER_URL + "/perception/object_detection_2d/detectron2/"):
        pass

    def eval(self):
        root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/"
        register_coco_instances("diesel_engine_val", {}, root+"val.json", root+"AugImages")
        self.cfg.DATASETS.TEST = ("diesel_engine_val",)
        self.cfg.TEST.EVAL_PERIOD = 100
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        evaluator = COCOEvaluator("diesel_engine_val", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, "diesel_engine_val")
        print(inference_on_dataset(self.trainer.model, val_loader, evaluator))

    def optimize(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()

    def reset(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()


def main():
    detectron2 = Detectron2Learner()
    json_file = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/CocoAugJSON.json"
    #image_root = input("image_root: ") # "../Dataset/AnnotatedImages/"
    image_root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/AugImages"
    # Import the dataset
    '''
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_root,
        labels_path=json_file)
    '''
    detectron2.fit()
    detectron2.eval()


if __name__ == '__main__':
	main()