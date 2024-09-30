import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import cv2
import random
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True,INPUT_SIZE: int = 224):
        self.is_train = is_train  # is_train 플래그를 저장
        self.INPUT_SIZE = INPUT_SIZE  # 입력 이미지 크기를 저장 
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(self.INPUT_SIZE, self.INPUT_SIZE),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            dropout_transform = A.CoarseDropout(
                max_holes=15, 
                max_height=int(0.1 * INPUT_SIZE), 
                max_width=int(0.1 * INPUT_SIZE), 
                fill_value=[random.randint(0, 127)] * 3,  # 0(검정)~128(회색) 사이 무작위 값을 선택하여 RGB 동일하게 적용
                p=0.5
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
                    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),  # 빈 공간을 흰색(255)으로 채움
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.VerticalFlip(p=0.2),
                    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5, border_mode=cv2.BORDER_CONSTANT, cval=255),  # 빈 공간을 흰색(255)으로 채움
                    A.ElasticTransform(alpha=1, sigma=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),  # 빈 공간을 흰색(255)으로 채움

                    dropout_transform,
                    A.Lambda(image=self.add_random_text, p=0.3),
                    A.Lambda(image=apply_dilation,p=0.4),
                    A.Lambda(image=apply_erosion,p=0.4),

                    # Noise and blur
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),

                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)
    
    def add_random_text(self, image: np.ndarray, **kwargs) -> np.ndarray: #증강용 함수, 이미지에 랜덤 텍스트를 추가한다.
        def random_word():
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            length = random.randint(3, 15)  # 4에서 12 사이의 길이로 랜덤 설정
            return ''.join(random.choice(letters) for _ in range(length))

        # 임의의 투명도 생성 함수
        def random_alpha():
            return random.uniform(0.5, 1.0)  # 0.3은 거의 투명, 1.0은 불투명

        # 임의의 폰트 크기 생성 함수
        def random_font_size():
            return random.uniform(0.3, 2)  # OpenCV에서는 폰트 크기를 스케일로 조정

        # 이미지 크기
        height, width, _ = image.shape

        # 텍스트를 그릴 위치들 (상단 및 하단에서만 랜덤 좌표 선택)
        top_y = random.randint(10, height // 5)  # 상단 영역의 랜덤 Y 좌표
        bottom_y = random.randint(height - height // 5, height - 10)  # 하단 영역의 랜덤 Y 좌표
        random_y = random.choice([top_y, bottom_y])

        # X 좌표는 이미지 폭에 따라 랜덤 설정
        random_x = random.randint(10, width - 100)

        # OpenCV 폰트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX


        # 랜덤 폰트, 색상, 투명도, 텍스트 추가
        random_text = random_word()
        color = (0,0,0)
        font_scale = random_font_size()  # 폰트 크기
        alpha = random_alpha()  # 투명도
        
        # 투명도를 적용한 텍스트 이미지 생성
        overlay = image.copy()
        cv2.putText(overlay, random_text, (random_x, random_y), font, font_scale, color, thickness=2, lineType=cv2.LINE_AA)
        
        # 알파 블렌딩으로 텍스트 투명도 조절
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)    
        return image
    
    def image_resize_with_padding(self, image): #이미지를 종횡비가 깨지지 않게, INPUT_SIZE*INPUT_SIZE Resize 및 Padding한다.
        h, w = image.shape[:2]
        if w > h:
            new_w = self.INPUT_SIZE
            new_h = int(h * (self.INPUT_SIZE / w))
        else:
            new_h = self.INPUT_SIZE
            new_w = int(w * (self.INPUT_SIZE / h))
    
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 패딩 추가
        top = (self.INPUT_SIZE - new_h) // 2
        bottom = self.INPUT_SIZE - new_h - top
        left = (self.INPUT_SIZE - new_w) // 2
        right = self.INPUT_SIZE - new_w - left
        
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        return padded_image
    
    def blackBackgorund_to_whiteBackground(self, image): #검은색 배경의 이미지라면 하얀색 배경의 이미지로 바꾼다.
        if np.mean(image) <= 127:
            image = 255 - image
        return image
    
    def enhance_and_binarize(self, image: np.ndarray, canny_threshold1: int = 100, canny_threshold2: int = 200, weight: float = 0.8) -> np.ndarray:
        """
        그레이스케일 이미지에 대해 평균값을 기반으로 픽셀을 강조하고
        Canny Edge를 적용하여 선을 강조한 뒤 다시 평균값을 기반으로 픽셀을 강조한다..
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 픽셀의 평균값 계산 후 평균보다 큰 픽셀들을 255로 설정
        mean_value = np.mean(gray_image)
        gray_image[gray_image > mean_value] = 255

        # Canny Edge를 이용해 선 강조
        edges = cv2.Canny(gray_image, threshold1=canny_threshold1, threshold2=canny_threshold2)
        enhanced_image = cv2.addWeighted(gray_image, weight, edges, 1 - weight, 0)

        # 픽셀의 평균값 계산 후 평균보다 큰 픽셀들을 255로 설정
        mean_value = np.mean(enhanced_image)
        enhanced_image[enhanced_image > mean_value] = 255

        return enhanced_image


    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        channel_diff = np.max(image, axis=-1) - np.min(image, axis=-1)
        if np.mean(channel_diff) > 50:  # 차이 값이 작으면 무채색이 많다고 판단
            image=self.image_resize_with_padding(image=image)
            transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
            return transformed['image']
        
        image = self.blackBackgorund_to_whiteBackground(image)
        
        image=self.enhance_and_binarize(image=image)
        image=self.image_resize_with_padding(image=image)

        graytorgb = np.stack([image] * 3, axis=-1)

        transformed = self.transform(image=graytorgb)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환
    
    
class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str, INPUT_SIZE: int = 224):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
            self.INPUT_SIZE = INPUT_SIZE
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        
        if self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train, INPUT_SIZE=self.INPUT_SIZE, WM_PROB=self.WM_PROB)
        
        return transform