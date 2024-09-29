import os  # 파일 경로 조작 및 디렉토리 생성
import torch  # PyTorch 기본 라이브러리
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 최적화 알고리즘
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training을 위한 모듈
from torch.utils.data import DataLoader  # 데이터 로딩을 위한 모듈
from sklearn.metrics import accuracy_score  # 정확도 계산을 위한 모듈
from tqdm.auto import tqdm  # 진행 상황을 시각적으로 보여주는 프로그레스 바
import wandb  # Weights and Biases (실험 추적 도구)


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        patience: int = 5,  # Early Stopping을 위한 patience 값 (개선되지 않는 에폭 수)
        min_delta: float = 0.0,  # 손실 감소 최소값 (이 값보다 더 개선되지 않으면 Early Stopping)
        use_amp: bool = True,
        model_name: str = "resnet18"
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = []  # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf')  # 가장 낮은 Loss를 저장할 변수
        self.patience = patience  # Early Stopping patience 값
        self.min_delta = min_delta  # 최소 손실 개선폭
        self.early_stopping_counter = 0  # Early Stopping을 위한 카운터
        self.use_amp = use_amp  # FP16 사용 여부
        self.scaler = GradScaler(enabled=use_amp)  # FP16 훈련을 위한 
        self.model_name = model_name    

    def save_model(self, epoch, loss, fold):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'{self.model_name}_fold_{fold}_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss == self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, f'{self.model_name}_fold_{fold}_best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch} epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets, img_paths in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)

            # 그레이스케일 이미지를 3채널로 변환
            if images.shape[1] == 1:  # 입력이 1채널인 경우
                images = images.repeat(1, 3, 1, 1)  # 1채널을 3채널로 복사
                
            self.optimizer.zero_grad()

            if self.use_amp:
                 # Mixed Precision 적용: autocast로 FP16 연산 영역을 설정
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, targets)

                # GradScaler를 사용하여 backward() 호출
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        incorrect_predictions = []  # 틀린 예측을 저장하는 리스트

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets, img_paths in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)

                # 그레이스케일 이미지를 3채널로 변환
                if images.shape[1] == 1:  # 입력이 1채널인 경우
                    images = images.repeat(1, 3, 1, 1)  # 1채널을 3채널로 복사

                if self.use_amp:
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, targets)
                else:
                    outputs = self.model(images)    
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()


                # 예측 결과 계산
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # 예측이 틀린 경우 기록
                for pred, target, img_path in zip(preds, targets, img_paths):
                    if pred != target:
                        incorrect_predictions.append({
                            "img_path": img_path,
                            "predicted": pred.item(),
                            "actual": target.item()
                        })

                progress_bar.set_postfix(loss=loss.item())
        
        # Accuracy 계산
        val_accuracy = accuracy_score(all_targets, all_preds)

        # WandB나 로그 파일로 틀린 예측 기록
        if incorrect_predictions:
            print(f"Incorrect Predictions: {len(incorrect_predictions)}")
            for item in incorrect_predictions:
                print(f"Image: {item['img_path']}, Predicted: {item['predicted']}, Actual: {item['actual']}")

        return total_loss / len(self.val_loader), val_accuracy

    def train(self, fold) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            print(f"fold : {fold}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # wandb에 학습 결과 기록
            try:
                wandb.log({"epoch": epoch+1, "fold":fold, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy":val_accuracy})
            except Exception as e:
                print(f"An error occurred during wandb logging: {e}")
            
            # Early Stopping 조건 확인
            if val_loss < self.lowest_loss:  # Validation Loss가 개선된 경우
                self.lowest_loss = val_loss  # 새로운 최저 validation loss 업데이트
                self.early_stopping_counter = 0  # 손실이 개선된 경우, 카운터 리셋
                print(f"Improvement in validation loss. Reset early stopping counter.")
            else:
                self.early_stopping_counter += 1  # 손실이 개선되지 않은 경우, 카운터 증가
                print(f"No improvement in validation loss. Early stopping counter: {self.early_stopping_counter}/{self.patience}")

            # 모델 저장
            self.save_model(epoch, val_loss, fold)
            
            # Early Stopping 조건을 만족하면 중단
            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

            # 학습률 스케줄러 갱신
        self.scheduler.step()  # val_loss가 아닌 epoch 기반으로 스케줄러 갱신

