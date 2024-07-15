import numpy as np


def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


if __name__ == "__main__":
    files = [
        r"D:\Projects\SVCFusion\data\train\pitch_aug_dict.npy",
        r"D:\Projects\SVCFusion\data\train2\pitch_aug_dict.npy",
        r"D:\Projects\SVCFusion\data\train3\pitch_aug_dict.npy",
        r"D:\Projects\SVCFusion\data\train4\pitch_aug_dict.npy",
        r"D:\Projects\SVCFusion\data\train5\pitch_aug_dict.npy",
        r"D:\Projects\SVCFusion\data\train6\pitch_aug_dict.npy",
        r"D:\Projects\SVCFusion\data\train7\pitch_aug_dict.npy",
    ]
    result = {}
    for file in files:
        file_dict = np.load(file, allow_pickle=True).all()
        result = merge_dicts(result, file_dict)
    np.save(r"D:\Projects\SVCFusion\data\pitch_aug_dict.npy", result)
