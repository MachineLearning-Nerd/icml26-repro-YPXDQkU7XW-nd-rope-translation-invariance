import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from timesformer.utils.parser import load_config, parse_args
from timesformer.models import build_model

def main():
    args = parse_args()
    cfg = load_config(args)

    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.DISTRIBUTED = False

    model = build_model(cfg)
    model.eval()
    model.cuda()

    B = 1
    C = 3
    T = cfg.DATA.NUM_FRAMES
    H = cfg.DATA.TRAIN_CROP_SIZE
    W = cfg.DATA.TRAIN_CROP_SIZE

    dummy_input = torch.randn(B, C, T, H, W).cuda()

    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    flops = FlopCountAnalysis(model, dummy_input)
    print("FLOPs (GFLOPs):", flops.total() / 1e9)
    print(parameter_count_table(model))

if __name__ == "__main__":
    main()