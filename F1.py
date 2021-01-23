import torch

def f1_loss(y_true, y_pred):
    '''Calculate F1 score. Can work with gpu tensors
    '''
    """if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        """
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    #f1.requires_grad = False
    return f1