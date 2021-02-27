
def dice_score(pred, true, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).astype("int")

    intersection = (pred*true).sum()
    dice = (2*intersection + eps)/((pred.sum() + true.sum()) + eps)

    return dice