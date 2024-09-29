import torch
import torch.nn as nn

class Loss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self, LABEL_SMOOTHING: bool = True, LABEL_SMOOTHING_VALUE: float = 0.1):
        super(Loss, self).__init__()
        self.LABEL_SMOOTHING = LABEL_SMOOTHING
        self.LABEL_SMOOTHING_VALUE = LABEL_SMOOTHING_VALUE
        
        if LABEL_SMOOTHING :
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_VALUE)  # 레이블 스무딩 적용
        else:
            self.loss_fn = nn.CrossEntropyLoss()


    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)