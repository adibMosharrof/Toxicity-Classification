
## Loss Design

### Factory

```python
import torch.nn as nn
from timm.loss import BinaryFocalLoss

class LossFactory:
    @staticmethod
    def create_loss(config):
        loss_cfg = config['model']['loss']
        loss_type = loss_cfg['type']
        params = loss_cfg['params']

        if loss_type == "focal":
            # timm provides a clean class implementation
            return BinaryFocalLoss(
                alpha=params.get('alpha', 0.25), 
                gamma=params.get('gamma', 2.0)
            )
        
        elif loss_type == "weighted_ce":
            # Standard PyTorch weighted BCE
            pos_weight = torch.tensor([params['pos_weight']])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        else:
            return nn.BCEWithLogitsLoss()
```


```python

import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class FocalLossWrapper(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        logits: (batch,) or (batch, 1) - raw logits
        targets: (batch,) - class indices (0 or 1)
        """
        # Ensure logits and targets are same shape for torchvision's implementation
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        
        return sigmoid_focal_loss(
            logits, 
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )
```


``` python
class ToxicityClassifier(nn.Module):
    def __init__(self, backbone, loss_fn, threshold: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn  # Injected from Factory
        self.threshold = threshold

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Flatten logits for binary loss functions
        logits = logits.view(-1)

        if labels is not None:
            # Most libraries expect labels as floats for binary tasks
            loss = self.loss_fn(logits, labels.float())
            return ModelOutput(loss=loss, logits=logits)
        
        # Inference logic
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).int()
        return preds, probs

```

Lets create a focallosswrapper using the focal loss from torchvision

Lets create a loss factory to select which loss to use.

In the config create a separate folder for the loss config.
Each loss will have its own config.

Then in the training we can select the loss in the defaults section.

I also have added example of how i want to redesign the model class.


