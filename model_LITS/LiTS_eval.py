import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    ResizeWithPadOrCropd
)

from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU
from monai.networks.nets import UNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

import logging
logging.getLogger("monai").setLevel(logging.ERROR)  # MONAI 로거의 레벨을 'ERROR'로 설정

import warnings
warnings.filterwarnings("ignore") # 경고제거용
#print_config()
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == True:
    print('cuda is available')


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["image"], a_min=-50, a_max=150, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 3.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)

import nibabel as nib
from monai.data import partition_dataset
nib.imageglobals.logger.setLevel(40)

data_dir = "D:\\LiTS\\data\\Task03_Liver"
split_json = "dataset_eval.json"

datasets = data_dir + split_json

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
test_files = load_decathlon_datalist(datasets, True, "test")

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_num=6,
    cache_rate=1.0,
    num_workers=4
)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
set_track_meta(False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=(128,128,64)
).to(device)

root_dir = 'D:\\severance\\models\\UNetr'

def validation(epoch_iterator_val, last_fc_size):
    
    
    model.load_state_dict(torch.load(os.path.join(root_dir, f"latest best_metric_model.pth")))
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            if val_labels.max() >= last_fc_size:
                val_labels[val_labels >= last_fc_size] = 0

            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (128, 128, 64), 2, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            if val_output_convert is not None and val_labels_convert is not None:

                dice_metric(y_pred=val_output_convert[:,2,...], y=val_labels_convert[:,2,...])
                surface_distance_metric(y_pred=val_output_convert[:,2,...], y=val_labels_convert[:,2,...])
                iou_metric(y_pred=val_output_convert[:,2,...], y=val_labels_convert[:,2,...])

            else:
                print("Invalid data encountered in validation")
                continue
        mean_dice_val = dice_metric.aggregate().item()
        
        mean_asd_val = surface_distance_metric.aggregate().item()
        mean_iou_val = iou_metric.aggregate().item()
        dice_metric.reset()
        
        surface_distance_metric.reset()
        iou_metric.reset()

    return mean_dice_val,  mean_asd_val, mean_iou_val
last_fc_size = 3
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
#hausdorff_distance_metric = HausdorffDistanceMetric(include_background=True, percentile=80)
surface_distance_metric = SurfaceDistanceMetric(include_background=False)
iou_metric = MeanIoU(include_background=False)

global_step = 0
dice_val_best = 0.0
global_step_best = 0  


epoch_loss_values = []
metric_values = []
hd_values = []
asd_values = []
iou_values = []


dice_val, asd_val, iou_val = validation(val_loader, last_fc_size)
metric_values.append(dice_val)
asd_values.append(asd_val)
iou_values.append(iou_val)


print(
        "Best Avg. Dice: {} \n  asd val: {} iou val:{}".format(dice_val, asd_val, iou_val)
    )
print(f'metric_values : {metric_values}, asd_values : {asd_values}, iou_values : {iou_values}')