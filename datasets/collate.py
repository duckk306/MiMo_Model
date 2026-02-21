import torch

def mimo_collate_fn(batch):
    """
    Custom collate_fn for MiMo multitask training.
    Allows None values for optional tasks.
    """
    images = []
    drive_areas = []
    detections = []

    for sample in batch:
        images.append(sample["image"])
        drive_areas.append(sample["drive_area"])
        detections.append(sample["detection"])  # keep None

    return {
        "image": torch.stack(images, dim=0),
        "drive_area": torch.stack(drive_areas, dim=0),
        "detection": detections  # list of None (OK)
    }
