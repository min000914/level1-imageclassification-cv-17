import os  # 파일 경로 조작을 위한 모듈
import cv2  # 이미지 처리 라이브러리
import pandas as pd  # 데이터 프레임을 다루기 위한 모듈
from typing import Tuple, Union, Callable  # 타입 힌팅을 위한 모듈
import torch  # PyTorch
from torch.utils.data import Dataset  # PyTorch 데이터셋 클래스

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable, is_inference: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 2:  # 그레이스케일 이미지의 경우
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.merge([image, image, image])

        image = self.transform(image)

        if self.is_inference:
            return image, img_path
        else:
            target = self.targets[index]
            return image, target, img_path

