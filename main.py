#!/usr/bin/env python
# coding: utf-8

# 필요한 모듈 import
from packages.Trainer import Trainer
from packages.Model import ModelSelector
from packages.Transform import TransformSelector
from packages.CustomDataset import CustomDataset
from packages.Loss import Loss

# 필요한 라이브러리 import
import os
import wandb
import random
import cv2
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from tqdm.auto import tqdm
from typing import Tuple, Any, Callable, List, Union
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

# --------------------------
# 환경 설정 및 초기화
# --------------------------

SERVER_NAME = "RTX4090"
os.chdir('/workspace' if SERVER_NAME == "RTX4090" else '/data/ephemeral/home/KJW')
print(f"현재 작업 디렉토리: {os.getcwd()}")

# WandB API 설정 및 로그인
os.environ['WANDB_API_KEY'] = 'wandb_api_key'
wandb.login()

# --------------------------
# 하이퍼파라미터 설정
# --------------------------

DATAPARALLEL = False  # 데이터 병렬처리 여부
USE_AMP = True  # FP16 적용 여부
BATCH_SIZE = 128  # 배치 크기
LEARNING_RATE = 0.0005  # 학습률
EPOCHS = 1  # 에폭 수
MODEL_TYPE = "timm"  # 모델 타입
MODEL_NAME = "resnet18"  # 모델 이름
ENSEMBLE_MODELS = [MODEL_NAME for _ in range(5)]  # 앙상블 모델
PRETRAINED = True  # 사전 학습 여부
INPUT_SIZE = 224  # 입력 이미지 사이즈
WM_PROB = 0.5
PATIENCE = 3  # 조기 종료 기준
LABEL_SMOOTHING = True  # 레이블 스무딩 사용 여부
LABEL_SMOOTHING_VALUE = 0.1  # 스무딩 값
GPU_NUM = "0"  # GPU 번호
device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")  # 장치 설정

# 경로 설정
SAVE_RESULT_PATH = "./train_result"  # 결과 저장 경로
TRAIN_DATA_DIR = "./data/train"  # 학습 데이터 경로
TRAIN_DATA_INFO_FILE = "./data/train.csv"  # 학습 데이터 CSV 경로
TEST_DATA_DIR = "./data/test"  # 테스트 데이터 경로
TEST_DATA_INFO_FILE = "./data/test.csv"  # 테스트 데이터 CSV 경로

# --------------------------
# 학습 준비 및 초기화
# --------------------------

# WandB 프로젝트 설정
wandb.init(project="BoostCamp-Project1", name=f"{SERVER_NAME}-{MODEL_NAME}-bs{BATCH_SIZE}-lr{LEARNING_RATE}")

# 랜덤 시드 설정
random_seed = 1111
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
wandb.config.update({"random_seed": random_seed})

# 학습 데이터 정보 로드
train_info = pd.read_csv(TRAIN_DATA_INFO_FILE)
num_classes = len(train_info['target'].unique())  # 클래스 수 확인

# 데이터 변환 (Transform) 선언
transform_selector = TransformSelector(transform_type="albumentations")
train_transform = transform_selector.get_transform(is_train=True)
val_transform = transform_selector.get_transform(is_train=False)
test_transform = transform_selector.get_transform(is_train=False)

# --------------------------
# 학습 과정
# --------------------------

# StratifiedKFold를 이용해 데이터를 나누고 학습을 진행
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed) # 교차 검증 설정

for fold, (train_idx, val_idx) in enumerate(skf.split(train_info, train_info['target'])):
    random_seed = random.randint(0, 1000000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"Training fold {fold+1}/{skf.get_n_splits()}")

    # 학습 및 검증 데이터셋 정의
    train_df, val_df = train_info.iloc[train_idx], train_info.iloc[val_idx]
    train_dataset = CustomDataset(root_dir=TRAIN_DATA_DIR, info_df=train_df, transform=train_transform)
    val_dataset = CustomDataset(root_dir=TRAIN_DATA_DIR, info_df=val_df, transform=val_transform)

    # DataLoader 선언
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # 모델 초기화 및 설정
    model_selector = ModelSelector(model_type=MODEL_TYPE, num_classes=num_classes, model_name=MODEL_NAME, pretrained=PRETRAINED)
    model = model_selector.get_model()
    if DATAPARALLEL:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # Optimizer 및 Scheduler 설정
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * 8, gamma=0.5)

    # 손실 함수 및 Trainer 설정
    loss_fn = Loss(LABEL_SMOOTHING=LABEL_SMOOTHING, LABEL_SMOOTHING_VALUE=LABEL_SMOOTHING_VALUE)
    trainer = Trainer(model=model, device=device, train_loader=train_loader, val_loader=val_loader,
                      optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, epochs=EPOCHS,
                      result_path=SAVE_RESULT_PATH, model_name=MODEL_NAME)

    # 학습 진행
    trainer.train(fold=fold)
    torch.cuda.empty_cache()  # GPU 메모리 정리

# --------------------------
# 추론(Inference)
# --------------------------

def ensemble_inference_save_each(
    model_name_list: List[str],  # 모델 이름 리스트
    device: torch.device, 
    test_loader: DataLoader, 
    save_result_path: str,  # 예측 값을 저장할 경로
    num_classes: int = 500,  # 총 클래스 수
    save_individual: bool = True  # 개별 모델의 출력을 저장할지 여부
):
    # 모델 예측값을 합산할 배열 초기화
    predictions = np.zeros((len(test_loader.dataset), num_classes))

    for fold, model_name in enumerate(model_name_list):
        # 모델 불러오기
        model_selector = ModelSelector(model_type='timm', num_classes=num_classes, model_name=model_name, pretrained=False)
        model = model_selector.get_model()

        if DATAPARALLEL:
            device_ids = [0, 1]  # 사용할 GPU 설정
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model_file = f"{model_name}_fold_{fold}_best_model.pt"
        model.load_state_dict(torch.load(os.path.join(SAVE_RESULT_PATH, model_file), map_location=device))
        model.to(device)
        model.eval()

        fold_predictions = []

        # 예측 수행
        with torch.no_grad():
            for images, _ in tqdm(test_loader):
                images = images.to(device)

                # 그레이스케일 이미지를 3채널로 변환
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

                # 모델 예측
                logits = model(images)
                logits = F.softmax(logits, dim=1)
                fold_predictions.append(logits.cpu().numpy())

        # 예측값을 배열로 변환 및 합산
        fold_predictions = np.vstack(fold_predictions)
        predictions += fold_predictions / len(model_name_list)

        # 개별 모델의 예측값을 CSV로 저장
        if save_individual:
            individual_output_file = os.path.join(save_result_path, f"{model_name}_fold_{fold}_output.csv")
            fold_df = pd.DataFrame(fold_predictions, columns=[f'class_{i}' for i in range(num_classes)])
            fold_df.to_csv(individual_output_file, index=False)

    # 최종 예측값 반환 (가장 높은 확률을 가진 클래스를 선택)
    return np.argmax(predictions, axis=1)


# 추론 데이터 설정
test_info = pd.read_csv(TEST_DATA_INFO_FILE)
test_dataset = CustomDataset(root_dir=TEST_DATA_DIR, info_df=test_info, transform=test_transform, is_inference=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# 앙상블에 사용할 모델 이름 리스트
model_name_list = ENSEMBLE_MODELS

# 추론 및 결과 저장
predictions = ensemble_inference_save_each(model_name_list, device, test_loader, SAVE_RESULT_PATH, num_classes)

# 현재 시간을 '년월일_시분초' 형식으로 추가
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

test_info['target'] = predictions
output_file = f"{SERVER_NAME}-{MODEL_NAME}-bs{BATCH_SIZE}-lr{LEARNING_RATE}-{current_time}.csv"
test_info.reset_index().rename(columns={"index": "ID"}).to_csv(output_file, index=False)

# WandB에 결과 업로드
wandb.save(output_file)
torch.cuda.empty_cache()  # 메모리 정리