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
from monai.networks.nets import SwinUNETR

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


root_dir = 'D:\\severance\\models\\SwinUnetr'
print(root_dir)
rand_num = np.random.randint(10)
print(rand_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == True:
    print('cuda is available')

import monai
print(monai.__version__)
###------------------------------------------Transforms------------------------------------------
num_samples = 2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        
        ScaleIntensityRanged(
            keys=["image"],
            #a_min=-175,
            #a_max=250,
            #실험적으로 수정
            #a_min=-100,
            #a_max=200,
            a_min=-50,
            a_max=150,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            #slice_thickness=3
            pixdim=(1.0, 1.0, 3.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(224, 224, 32),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        #RandCropByPosNegLabeld에서 smaller=True하면 이것도 해야함, spatial_size divisible by 32주의
        ResizeWithPadOrCropd(keys=["image", "label"],
        spatial_size=(224, 224, 32),
        mode='constant')
    ]
)


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
'''RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        ResizeWithPadOrCropd(keys=["image", "label"],
        spatial_size=(96, 96, 96),
        mode='constant'),'''
###----------------------------------------NII header debug--------------------------
'''import nibabel as nib

# nii header confirm
data_file_btcv = 'D:\data1\imagesTr\img0023.nii.gz'
data_file_severance = 'D:\severance\imagesTr\\CT1.nii.gz'
data_b = nib.load(data_file_btcv)
data_s = nib.load(data_file_severance)


print('btcv\n',data_b.shape,'\n',data_b.header,'\n------------------------')
print('severance\n',data_s.shape,'\n',data_s.header,'\n------------------------')'''

###---------------------------------------pixdim error solving------------------------------
import nibabel as nib
from monai.data import partition_dataset
nib.imageglobals.logger.setLevel(40)

data_dir = "D:\\severance\\new_data\\"
split_json = "dataset_1.json"

datasets = data_dir + split_json

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
test_files = load_decathlon_datalist(datasets, True, "test")
#train_files, val_files = partition_dataset(datalist, ratios=[0.8, 0.2], shuffle=False)

# 학습 데이터셋 및 데이터 로더
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=12,
    cache_rate=1.0,
    num_workers=8
)
train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=2, shuffle=True)

# 검증 데이터셋 및 데이터 로더
val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_num=6,
    cache_rate=1.0,
    num_workers=4
)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

# 테스트 데이터셋 및 데이터 로더
test_ds = CacheDataset(
    data=test_files,
    transform=val_transforms,
    cache_num=6,
    cache_rate=1.0,
    num_workers=4
)
test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)


set_track_meta(False)




###-------------------------------------------Modeling------------------------------------------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=(224,224,32),
    in_channels=1,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True,).to(device)

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

def validation(epoch_iterator_val, last_fc_size):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if val_labels.max() >= last_fc_size:
                val_labels[val_labels >= last_fc_size] = 0
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (224, 224, 32), 2, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            if val_output_convert is not None and val_labels_convert is not None:
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                
                surface_distance_metric(y_pred=val_output_convert, y=val_labels_convert)
                iou_metric(y_pred=val_output_convert, y=val_labels_convert)
            else:
                print("Invalid data encountered in validation")
                continue
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        
        mean_asd_val = surface_distance_metric.aggregate().item()
        mean_iou_val = iou_metric.aggregate().item()
        dice_metric.reset()
        
        surface_distance_metric.reset()
        iou_metric.reset()

    return mean_dice_val,  mean_asd_val, mean_iou_val


def test(epoch_iterator_test, last_fc_size):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        # tqdm을 사용하여 테스트 진행 상태를 시각적으로 표시
        epoch_iterator_test = tqdm(epoch_iterator_test, desc="Testing", dynamic_ncols=True)
        for batch in epoch_iterator_test:
            test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
            if test_labels.max() >= last_fc_size:
                test_labels[test_labels >= last_fc_size] = 0
            with torch.cuda.amp.autocast():
                test_outputs = sliding_window_inference(test_inputs, (224, 224, 32), 2, model)
            test_labels_list = decollate_batch(test_labels)
            test_labels_convert = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
            test_outputs_list = decollate_batch(test_outputs)
            test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]
            if test_output_convert is not None and test_labels_convert is not None:
                dice_metric(y_pred=test_output_convert, y=test_labels_convert)
                
                surface_distance_metric(y_pred=test_output_convert, y=test_labels_convert)
                iou_metric(y_pred=test_output_convert, y=test_labels_convert)
            else:
                print("Invalid data encountered in test")
                continue
            epoch_iterator_test.set_description("Test (%d / %d Steps)" % (global_step, 10.0))
        
        mean_dice_test = dice_metric.aggregate().item()
        
        mean_asd_test = surface_distance_metric.aggregate().item()
        mean_iou_test = iou_metric.aggregate().item()
        
        dice_metric.reset()
        
        surface_distance_metric.reset()
        iou_metric.reset()

    return mean_dice_test,  mean_asd_test, mean_iou_test

def train(global_step, train_loader, dice_val_best, global_step_best, last_fc_size):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        #print('mmmmmmmmmmmmmmmmmmmmmmmmmm',np.shape(x))
        #x shape = torch.Size([4, 1, 96, 96, 96])
        
        if y.max() >= last_fc_size:
            print("\nReplace labels >= {} to 0".format(last_fc_size))
            y[y >= last_fc_size] = 0
        #squeeze and convert 정수 -> weighted ce에 적합한 형태는 정수라벨임
        
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            if logit_map is not None and y is not None:
                loss = loss_function(logit_map, y)
            else:
                print("Invalid data encountered in training")
                continue  # 다음 배치로 넘어감
        scaler.scale(loss).backward()


        total_norm=0
        for p in model.parameters():
            
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if global_step%500 == 0:
            
            print(total_norm)
        #if (global_step >= 1000 and total_norm>=200000):
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=300000)
        #    print('clip grad over 20000!')
        #    print(total_norm)
        epoch_loss += loss.item()
        
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
        

        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
        
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val, asd_val, iou_val = validation(epoch_iterator_val, last_fc_size)
            epoch_loss /= step

            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            
            asd_values.append(asd_val)
            iou_values.append(iou_val)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, f"{rand_num} best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} \n  asd val: {} iou val:{}".format(dice_val_best, dice_val, asd_val, iou_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} \n asd val: {} iou val:{}".format(dice_val_best, dice_val, asd_val, iou_val)
                )
        if (global_step % (eval_num*3) == 0 and global_step != 0) or global_step == max_iterations:
            dice_test, asd_test, iou_test = test(test_loader, last_fc_size)
            test_metric_values.append(dice_test)
            
            test_asd_values.append(asd_test)
            test_iou_values.append(iou_test)
            #print(test_metric_values)
            print(
                    "Test: Current Avg. Dice: {} \n asd test: {} iou test:{}".format(dice_test, asd_test, iou_test)
                )

        global_step += 1
    return global_step, dice_val_best, global_step_best


###------------------------------------Training---------------------------------------
#내 y가 0,1,2를 넘는 경우를 pass하고 훈련진행하기위한 장치
last_fc_size = 3
max_iterations = 30000
eval_num = 500
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
#hausdorff_distance_metric = HausdorffDistanceMetric(include_background=True, percentile=80)
surface_distance_metric = SurfaceDistanceMetric(include_background=True)
iou_metric = MeanIoU(include_background=True)

global_step = 0
dice_val_best = 0.0
global_step_best = 0  


epoch_loss_values = []
metric_values = []
hd_values = []
asd_values = []
iou_values = []

test_metric_values = []
test_hd_values = []
test_asd_values = []
test_iou_values = []

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best, last_fc_size)
print('epoch_loss_values:',epoch_loss_values, 'metric_values:',metric_values, 'test_metric_values:',test_metric_values)

import json

# Data to be saved
data = {
    "epoch_loss_values": epoch_loss_values,
    "metric_values": metric_values,
    "asd_values": asd_values,
    "iou_values": iou_values,
    "test_metric_values": test_metric_values,
    "test_hd_values": test_hd_values,
    "test_asd_values": test_asd_values,
    "test_iou_values": test_iou_values
}

# Convert to JSON string
json_data = json.dumps(data, indent=4)

# Write to file
with open('D:\\severance\\models\\SwinUnetr\\metrics.json', 'w') as file:
    file.write(json_data)

###------------------------------------Plot Loss, metric------------------------------------------------

'''
with open('D:\\severance\\SwinUnetr\\metrics.json', 'r') as file:
    json_data = file.read()

data = json.loads(json_data)
epoch_loss_values = data["epoch_loss_values"]
metric_values = data["metric_values"]
asd_values = data["asd_values"]
iou_values = data["iou_values"]
test_metric_values = data["test_metric_values"]
test_hd_values = data["test_hd_values"]
test_asd_values = data["test_asd_values"]
test_iou_values = data["test_iou_values"]
'''

plt.figure("Performance", (18, 6))

# Plot for iteration average loss
plt.subplot(1, 4, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
plt.xlabel("Iteration")
plt.plot(x, epoch_loss_values, label='Loss')
plt.legend()

# Plot for validation mean dice
plt.subplot(1, 4, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
plt.xlabel("Iteration")
plt.plot(x, metric_values, label='Dice')
plt.legend()

# Plot for additional metrics
plt.subplot(1, 4, 3)
plt.title("Validation Metrics")
x = [eval_num * (i + 1) for i in range(len(asd_values))]
plt.xlabel("Iteration")
plt.plot(x, asd_values, label='Average Surface Distance')
plt.plot(x, iou_values, label='IoU')
plt.legend()

# Plot for test metrics
plt.subplot(1, 4, 4)
plt.title("Test Metrics")
x = [eval_num * 10 * (i + 1) for i in range(len(test_metric_values))]
plt.plot(x, test_metric_values, label='Test Dice')
plt.plot(x, test_asd_values, label='Test ASD')
plt.plot(x, test_iou_values, label='Test IoU')
plt.xlabel("Iteration")
plt.legend()

plt.show()
loss_graph = "D:\\severance\\models\\SwinUnetr\\loss_graph"
if not os.path.exists(loss_graph):
    os.makedirs(loss_graph)
    
    plt.savefig(os.path.join(loss_graph, 'loss_graph'))
    plt.clf()
###-------------------------------------------------Check best model-------------------------------------------

        
###-----------val result save------------------

model.load_state_dict(torch.load(os.path.join(root_dir, f"{rand_num} best_metric_model.pth")))
model.eval()

with torch.no_grad():
    for num in range(20):
        img = val_ds[num]["image"]
        print(img.shape)
        img = torch.unsqueeze(img, 1).cuda()
        print(img.shape)
        out = sliding_window_inference(img, (224, 224, 32), 4, model, overlap=0.8)
        print(out.shape)
        
        out = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.uint8)
        out = out.squeeze(0)
        print(out.shape)
        
        img = nib.Nifti1Image(out, affine=np.eye(4))
        
        label = val_ds[num]["label"].squeeze().cpu().numpy().astype(np.uint8)
        label = nib.Nifti1Image(label, affine=np.eye(4))

        nib.save(img,f"D:\\severance\\models\\SwinUnetr\\val_result\\{str(num)}-output.nii.gz")
        nib.save(label,f"D:\\severance\\models\\SwinUnetr\\val_result\\{str(num)}-label.nii.gz")


###-----------test result save--------------------
model.load_state_dict(torch.load(os.path.join(root_dir, f"{rand_num} best_metric_model.pth")))
model.eval()

with torch.no_grad():
    for num in range(19):
        img = test_ds[num]["image"]
        img = torch.unsqueeze(img, 1).cuda()
        out = sliding_window_inference(img, (224, 224, 32), 4, model, overlap=0.8)

        out = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.uint8)
        out = out.squeeze(0)
        img = nib.Nifti1Image(out, affine=np.eye(4))
        
        label = val_ds[num]["label"].squeeze().cpu().numpy().astype(np.uint8)
        label = nib.Nifti1Image(label, affine=np.eye(4))

        nib.save(img,f"D:\\severance\\models\\SwinUnetr\\test_result\\{str(num)}-output.nii.gz")
        nib.save(label,f"D:\\severance\\models\\SwinUnetr\\test_result\\{str(num)}-label.nii.gz")

slice_map = {"CT395.nii.gz":25, "CT396.nii.gz" : 20,"CT397.nii.gz" : 30, "CT398.nii.gz":30, "CT399.nii.gz":30}
case_num = 5
model.load_state_dict(torch.load(os.path.join(root_dir, f"{rand_num} best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for num in range(case_num):
        img = val_ds[num]["image"]
        label = val_ds[num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(val_inputs, (224, 224, 32), 4, model, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 40], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 40])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 40])
        plt.show()

##Clean up dir
#if directory is None:
#    shutil.rmtree(root_dir)
###=============================================Inference check==============================
slice_map = {"CT405.nii.gz":25, "CT406.nii.gz" : 20,"CT407.nii.gz" : 30, "CT408.nii.gz":30, "CT409.nii.gz":30}  # 예시 슬라이스 맵 (실제 파일명에 맞게 수정 필요)
test_case_num = 5  # 테스트 케이스의 수

model.load_state_dict(torch.load(os.path.join(root_dir, f"{rand_num} best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for num in range(test_case_num):
        img = test_ds[num]["image"]
        label = test_ds[num]["label"]
        test_inputs = torch.unsqueeze(img, 1).cuda()
        test_labels = torch.unsqueeze(label, 1).cuda()
        test_outputs = sliding_window_inference(test_inputs, (224, 224, 32), 4, model, overlap=0.8)
        
        plt.figure("check_test", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Test Image")
        plt.imshow(test_inputs.cpu().numpy()[0, 0, :, :, 40], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Test Label")
        plt.imshow(test_labels.cpu().numpy()[0, 0, :, :, 40])
        plt.subplot(1, 3, 3)
        plt.title("Test Output")
        plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, 40])
        plt.show()