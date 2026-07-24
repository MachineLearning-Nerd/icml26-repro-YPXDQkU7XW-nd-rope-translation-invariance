import os
import torch
import numpy as np
import importlib
import logging
from tqdm import tqdm
from dataset import PartNormalDataset
import hydra
import omegaconf

# ========== Part labels ==========
seg_classes = {
    'Earphone': [16, 17, 18],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11],
    'Laptop': [28, 29],
    'Cap': [6, 7],
    'Skateboard': [44, 45, 46],
    'Mug': [36, 37],
    'Guitar': [19, 20, 21],
    'Bag': [4, 5],
    'Lamp': [24, 25, 26, 27],
    'Table': [47, 48, 49],
    'Airplane': [0, 1, 2, 3],
    'Pistol': [38, 39, 40],
    'Chair': [12, 13, 14, 15],
    'Knife': [22, 23]
}

seg_label_to_cat = {}
for cat in seg_classes:
    for l in seg_classes[cat]:
        seg_label_to_cat[l] = cat


def to_categorical(y, num_classes):
    """ one-hot encode """
    return torch.eye(num_classes)[y.cpu().numpy()].cuda()


@hydra.main(config_path='config', config_name='partseg_ndrope')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"TEST-N{args.num_point}")

    # ========= Dataset =========
    root = hydra.utils.to_absolute_path(
        'data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    )

    TEST_DATASET = PartNormalDataset(
        root=root,
        npoints=args.num_point,
        split='test',
        normal_channel=args.normal
    )

    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )

    # ========= Model =========
    args.input_dim = (6 if args.normal else 3) + 16
    args.num_class = 50
    num_category = 16
    num_part = args.num_class

    model_module = importlib.import_module(
        f'models.{args.model.name}.model'
    )
    classifier = model_module.PointTransformerSeg(args).cuda()

    checkpoint = torch.load('best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    logger.info("Loaded best_model.pth")
    logger.info("=" * 50)
    logger.info(f"Evaluating with num_point = {args.num_point}")
    logger.info("=" * 50)

    # ========= Metrics =========
    total_correct = 0
    total_seen = 0
    total_seen_class = [0] * num_part
    total_correct_class = [0] * num_part
    shape_ious = {cat: [] for cat in seg_classes}

    # ========= Evaluation =========
    with torch.no_grad():
        for points, label, target in tqdm(test_loader):
            B, N, _ = points.shape
            points = points.float().cuda()
            label = label.long().cuda()
            target = target.long().cuda()

            seg_pred = classifier(
                torch.cat([
                    points,
                    to_categorical(label, num_category).repeat(1, N, 1)
                ], -1)
            )

            seg_pred = seg_pred.cpu().numpy()
            target = target.cpu().numpy()

            pred = np.zeros((B, N), dtype=np.int32)

            for i in range(B):
                cat = seg_label_to_cat[target[i, 0]]
                logits = seg_pred[i][:, seg_classes[cat]]
                pred[i] = np.argmax(logits, axis=1) + seg_classes[cat][0]

            total_correct += np.sum(pred == target)
            total_seen += B * N

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += np.sum((pred == l) & (target == l))

            for i in range(B):
                cat = seg_label_to_cat[target[i, 0]]
                part_ious = []
                for l in seg_classes[cat]:
                    I = np.sum((pred[i] == l) & (target[i] == l))
                    U = np.sum((pred[i] == l) | (target[i] == l))
                    part_ious.append(1.0 if U == 0 else I / float(U))
                shape_ious[cat].append(np.mean(part_ious))

    # ========= Final Metrics =========
    all_shape_ious = []
    for cat in shape_ious:
        all_shape_ious.extend(shape_ious[cat])
        shape_ious[cat] = np.mean(shape_ious[cat])

    overall_acc = total_correct / float(total_seen)
    class_avg_acc = np.mean(
        np.array(total_correct_class) / np.maximum(np.array(total_seen_class), 1)
    )
    class_avg_iou = np.mean(list(shape_ious.values()))
    instance_avg_iou = np.mean(all_shape_ious)

    # ========= Print =========
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" Test Results (num_point = {args.num_point})")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy      : {overall_acc:.6f}")
    logger.info(f"Class Avg Accuracy    : {class_avg_acc:.6f}")

    for cat in sorted(shape_ious):
        logger.info(f"mIoU of {cat:<12}: {shape_ious[cat]:.6f}")

    logger.info(f"Class Avg mIoU        : {class_avg_iou:.6f}")
    logger.info(f"Instance Avg mIoU     : {instance_avg_iou:.6f}")


if __name__ == '__main__':
    main()