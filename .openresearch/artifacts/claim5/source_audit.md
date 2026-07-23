# Source audit

Paper anchors: Table 3 and Section 5.1 identify the 85.97 result as ModelNet40
instance-average mIoU. The pinned executable source instead names
`shapenetcore_partanno_segmentation_benchmark_v0_normal`, constructs
`PartNormalDataset`, and computes ShapeNetPart part IoU. The 154-file release
also has no SemanticKITTI file.
