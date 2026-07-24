# Method

Inventory and hash all 154 upstream files. Trace the training and evaluation
entrypoints from dataset path through loader, category/label cardinalities, and
metric aggregation. The independent verifier requires the pinned commit,
ShapeNetPart path, absence of ModelNet loader use in the entrypoints, 16
categories, 50 parts, and zero checkpoints.

Negative control: the generic `ModelNetDataLoader` definition exists in the
shared dataset module, so mere string presence must not trigger falsification;
only executable entrypoint use counts.
