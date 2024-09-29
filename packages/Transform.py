import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import cv2
import random
import string
from augraphy import WaterMark

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True, INPUT_SIZE: int = 224, WM_PROB: float = 0.5):
        self.is_train = is_train  # is_train 플래그를 저장
        self.INPUT_SIZE = INPUT_SIZE  # 입력 이미지 크기를 저장 
        self.WM_PROB = WM_PROB  # 워터마크 확률을 저장

        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(INPUT_SIZE, INPUT_SIZE,interpolation=cv2.INTER_CUBIC),  # 이미지를 INPUT_SIZExINPUT_SIZE 크기로 리사이즈 (모델 입력에 맞춤)
            A.Normalize(mean=[0.5], std=[0.5]),  # 그레이스케일 이미지의 정규화 (mean=0.5, std=0.5)
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # CoarseDropout 여러 번 반복하는 변환 설정
            dropout_transforms = []
            for _ in range(10):
                max_height = random.randint(int(0.05 * INPUT_SIZE), int(0.15 * INPUT_SIZE))
                max_width = random.randint(int(0.05 * INPUT_SIZE), int(0.15 * INPUT_SIZE))
                fill_value = random.randint(0, 255)
                dropout_transforms.append(
                    A.CoarseDropout(max_holes=random.randint(1, 2),
                                    max_height=max_height,
                                    max_width=max_width,
                                    fill_value=fill_value, p=0.5)
                )      

            # OpenCV 기반의 Erosion 및 Dilation 함수 정의
            def apply_erosion(img, **params):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                return cv2.erode(img, kernel, iterations=1)

            def apply_dilation(img, **params):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                return cv2.dilate(img, kernel, iterations=1)

            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    # Geometric transformations
                     A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),  # 빈 공간을 흰색(255)으로 채움
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    # A.VerticalFlip(p=0.5),
                    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5, border_mode=cv2.BORDER_CONSTANT, cval=255),  # 빈 공간을 흰색(255)으로 채움
                    A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),  # 빈 공간을 흰색(255)으로 채움

                    *dropout_transforms,

                    # Morphological transformations using OpenCV
                    A.Lambda(image=apply_erosion, p=0.5),
                    A.Lambda(image=apply_dilation, p=0.5),

                    # Noise and blur
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),

                    # Sketch-specific augmentations
                    A.CoarseDropout(max_holes=20, max_height=int(0.03 * INPUT_SIZE), max_width=int(0.03 * INPUT_SIZE), fill_value=255, p=0.5),
                    A.CoarseDropout(max_holes=20, max_height=int(0.03 * INPUT_SIZE), max_width=int(0.03 * INPUT_SIZE), fill_value=0, p=0.5),
                    A.CoarseDropout(max_holes=10, max_height=int(0.05 * INPUT_SIZE), max_width=int(0.05 * INPUT_SIZE), fill_value=255, p=0.5),
                    A.CoarseDropout(max_holes=10, max_height=int(0.05 * INPUT_SIZE), max_width=int(0.05 * INPUT_SIZE), fill_value=0, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=int(0.15 * INPUT_SIZE), max_width=int(0.15 * INPUT_SIZE), fill_value=0, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=int(0.15 * INPUT_SIZE), max_width=int(0.15 * INPUT_SIZE), fill_value=255, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=int(0.15 * INPUT_SIZE), max_width=int(0.15 * INPUT_SIZE), fill_value=0, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=int(0.15 * INPUT_SIZE), max_width=int(0.15 * INPUT_SIZE), fill_value=255, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=int(0.15 * INPUT_SIZE), max_width=int(0.15 * INPUT_SIZE), fill_value=0, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=int(0.15 * INPUT_SIZE), max_width=int(0.15 * INPUT_SIZE), fill_value=255, p=0.5),
                                                            

                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def get_random_text(self, length):
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for _ in range(length))

    def add_watermark(self, image: np.ndarray) -> np.ndarray:
        # Augraphy의 WaterMark 클래스를 사용하여 워터마크 생성
        text_length = random.randint(5, 15)
        font_size = random.randint(1, 15)
        font_thickness = random.randint(1, 15)
        watermark_locations = ["center", "top-left", "top-right", "bottom-left", "bottom-right"]
        random_text = self.get_random_text(text_length)
        # print(random_text)
        watermark = WaterMark(
            watermark_word=random_text,
            watermark_font_size=(font_size, font_size),
            watermark_font_thickness=(font_thickness, font_thickness),
            watermark_font_type=cv2.FONT_HERSHEY_SIMPLEX,
            watermark_rotation=(0, 360),
            watermark_location=random.choice(watermark_locations),
            watermark_color=(125, 125, 125),
            watermark_method="light"
        )

        # 워터마크를 투명 배경으로 생성 (배경이 없는 이미지에 적용)
        img_height, img_width = image.shape[:2]
        img_with_watermark = np.full((img_height, img_width,3), 255, dtype="uint8")

        # 워터마크 생성 및 적용
        img_with_watermark = watermark(img_with_watermark)

        # 워터마크 이미지를 그레이스케일로 변환
        img_with_watermark_gray = cv2.cvtColor(img_with_watermark, cv2.COLOR_RGB2GRAY)

        # 투명도 설정 (0.3 ~ 0.7 사이의 랜덤 값)
        alpha = random.uniform(0.1, 0.7) # alpha 값이 0에 가까울수록 워터마크가 거의 보이지 않으며, 원본 이미지가 더 많이 보입니다.

        # 원본 이미지에 알파 블렌딩 방식으로 워터마크 적용
        overlay = np.zeros_like(image, dtype=np.uint8)

        # 워터마크를 이미지 중앙에 배치
        wm_height, wm_width = img_with_watermark_gray.shape[:2]
        top_y = (img_height - wm_height) // 2  # Y축 중앙에 정렬
        left_x = (img_width - wm_width) // 2  # X축 중앙에 정렬

        # 오버레이 이미지에 워터마크를 배치할 위치에 설정
        overlay[top_y:top_y + wm_height, left_x:left_x + wm_width] = img_with_watermark_gray

        # 원본 이미지와 오버레이 이미지(워터마크)를 알파 블렌딩으로 합성
        blended_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return blended_image


    def pad_to_square(self, image: np.ndarray) -> np.ndarray:
        # 이미지의 가로와 세로 중 큰 값을 찾기
        height, width = image.shape[:2]
        size = max(height, width)
        
        # 이미지가 그레이스케일일 경우와 RGB일 경우를 구분하여 패딩 적용
        if len(image.shape) == 2:
            # 흑백 이미지의 경우 1채널 적용
            padded_image = np.full((size, size), 255, dtype=np.uint8)
        else:
            # 컬러 이미지의 경우 3채널 적용
            padded_image = np.full((size, size, 3), 255, dtype=np.uint8)
        
        # 원본 이미지를 패딩한 이미지의 중앙에 배치
        y_offset = (size - height) // 2
        x_offset = (size - width) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
        
        return padded_image

    def apply_grayscale(self, image: np.ndarray) -> np.ndarray:
        # 이미지를 그레이스케일로 변환
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return grayscale_image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
        return enhanced_image

    def visualize(self, image, title):
        if image.shape[0] == 1:  # 이미지가 (1, H, W) 형태인 경우
            image = image.squeeze(0)  # 첫 번째 차원을 제거하여 (H, W)로 변경
        plt.imshow(image, cmap='gray')  # 그레이스케일 이미지로 시각화
        plt.title(title)
        plt.axis('off')
        plt.show()  # 각 이미지 출력 후 별도로 시각화

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 1. 패딩을 통해 이미지를 정사각형으로 변환 (사이즈 불균형 해결)
        image = self.pad_to_square(image)
        
        # 2. Grayscale 변환 적용
        grayscale_image = self.apply_grayscale(image)

        # 3. 훈련 데이터일 경우에만 워터마크를 50% 확률로 추가
        if self.is_train and random.random() < self.WM_PROB:
            grayscale_image = self.add_watermark(grayscale_image)
    
        # 4. Albumentations 변환 적용 및 결과 반환 (랜덤 변환 및 정규화)
        transformed = self.transform(image=grayscale_image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환



class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str, INPUT_SIZE: int = 224, WM_PROB: float = 0.5):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
            self.INPUT_SIZE = INPUT_SIZE
            self.WM_PROB = WM_PROB
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        
        if self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train, INPUT_SIZE=self.INPUT_SIZE, WM_PROB=self.WM_PROB)
        
        return transform

