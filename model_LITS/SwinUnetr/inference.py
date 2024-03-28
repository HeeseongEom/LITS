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
from monai.networks.nets import SwinUNETR_ori

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


root_dir = 'D:\\Tumor_Segmentation\\LiTS\\models\\SwinUnetr'
print(root_dir)
rand_num = np.random.randint(10)
print(rand_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == True:
    print('cuda is available')

import monai
print(monai.__version__)
###------------------------------------------Transforms------------------------------------------
num_samples = 1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 64),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        ResizeWithPadOrCropd(keys=["image", "label"],
        spatial_size=(128, 128, 64),
        mode='constant')
    ]
)

import nibabel as nib
from monai.data import partition_dataset
nib.imageglobals.logger.setLevel(40)

data_dir = "D:\\Tumor_Segmentation\\LiTS\\data\\Task03_Liver\\"
split_json = "dataset_eval.json"



datasets = data_dir + split_json

val_files = load_decathlon_datalist(datasets, True, "validation")
#train_files, val_files = partition_dataset(datalist, ratios=[0.8, 0.2], shuffle=False)



# 검증 데이터셋 및 데이터 로더
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



model = SwinUNETR_ori(
    img_size=(128, 128, 64),
    in_channels=1,
    out_channels=3,
    feature_size=48
).to(device)

#weight = torch.load("D:\\severance\\model_swinvit.pt")
#model.load_from(weights=weight)
#print("Using pretrained self-supervied Swin UNETR backbone weights !")

torch.backends.cudnn.benchmark = True

#loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#class_weights = torch.tensor([1, 2, 3], dtype=torch.float).cuda()  # 클래스 0과 1은 가중치 1, 클래스 2는 가중치 2

# 교차 엔트로피 손실 함수 정의
#loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

scaler = torch.cuda.amp.GradScaler()


###------------------------------------Train/Validation define-------------------------------------------

def model_predictor(inputs):
    # model을 호출하여 logit과 hidden_states를 받음
    logit, hidden_states = model(inputs)
    # sliding_window_inference 함수에는 logit만 필요하므로 logit만 반환
    return logit

def merge_channels(data, method='max'):
    """
    data: 입력 데이터, 형태는 [B, C, H, W, D]
    method: 'max' 또는 'mean', 채널을 통합하는 방식
    """
    if method == 'max':
        merged = data.max(dim=1)[0]  # 모든 채널에서 최대값
    elif method == 'mean':
        merged = data.mean(dim=1)  # 모든 채널의 평균
    else:
        raise ValueError("Unsupported merge method. Use 'max' or 'mean'.")
    return merged

def plot_feature_maps(data, slice_index):
    feature_map = data[0, :, :, slice_index].cpu().numpy()
    plt.imshow(feature_map, cmap='gray')
    plt.title(f'Feature Map - Slice {slice_index}')
    plt.colorbar()
    plt.show()

def validation(epoch_iterator_val, last_fc_size):
    model.load_state_dict(torch.load(os.path.join(root_dir, f"latest best_metric_model.pth")))
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            print(val_inputs.shape)
            #val_labels=torch.round(val_labels)
            if val_labels.max() >= last_fc_size:
                val_labels[val_labels >= last_fc_size] = 0
            with torch.cuda.amp.autocast():
                logit, hidden_states = model(val_inputs)
                print(len(hidden_states), type(hidden_states), hidden_states[0].shape, hidden_states[1].shape,hidden_states[2].shape,hidden_states[3].shape,hidden_states[4].shape)
                val_outputs = sliding_window_inference(val_inputs, (128, 128, 64), 1, model_predictor)
            for i in range(len(hidden_states)):

                hidden_states[i]=merge_channels(hidden_states[i], method='mean')
                shape = hidden_states[i].shape
                plot_feature_maps(hidden_states[i], slice_index=shape[3]//2)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            if val_output_convert is not None and val_labels_convert is not None:
                #print("y_pred shape:", [val_pred_tensor.shape for val_pred_tensor in val_output_convert])
                #print("y shape:", [val_label_tensor.shape for val_label_tensor in val_labels_convert])
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                surface_distance_metric(y_pred=val_output_convert, y=val_labels_convert)
                iou_metric(y_pred=val_output_convert, y=val_labels_convert)

            else:
                print("Invalid data encountered in validation")
                continue
            
        mean_dice_val = dice_metric.aggregate().item()
        
        mean_asd_val = surface_distance_metric.aggregate().item()
        mean_iou_val = iou_metric.aggregate().item()
        dice_metric.reset()
        
        surface_distance_metric.reset()
        iou_metric.reset()
        print(mean_dice_val, mean_asd_val, mean_iou_val)

    return mean_dice_val, mean_asd_val, mean_iou_val



###------------------------------------Training---------------------------------------
#내 y가 0,1,2를 넘는 경우를 pass하고 훈련진행하기위한 장치
last_fc_size = 3

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
#hausdorff_distance_metric = HausdorffDistanceMetric(include_background=True, percentile=80)
surface_distance_metric = SurfaceDistanceMetric(include_background=True)
iou_metric = MeanIoU(include_background=True)

global_step = 0
dice_val_best = 0.0
global_step_best = 0

#예제 llm feature

epoch_loss_values = []
metric_values = []
hd_values = []
asd_values = []
iou_values = []

dice, asd, iou = validation(val_loader, 3)
#print(dice, asd, iou)


import json

# Data to be saved
data = {
    "epoch_loss_values": epoch_loss_values,
    "metric_values": metric_values,
    "asd_values": asd_values,
    "iou_values": iou_values
}

