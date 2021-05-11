import argparse

str2bool = lambda arg: arg.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Evaluate a model for surgical instrument anticipation.")
parser.register('type', 'bool', str2bool)

parser.add_argument('--sample_path', type=str, default='/home/kun/Desktop/miccai21/resnet_pretrain_lightning/logs/210503-143402_FeatureExtraction_Cholec80FeatureExtract_cnn_TwoHeadResNet50Model_multitask_ii/cholec80_pickle_export_5/1.0fps/')
parser.add_argument('--out_dir', type=str, default='./horizon5/')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--uncertainty_type', type=str, default='aleatoric_cls', help='options: (epistemic_reg | epistemic_cls | aleatoric_cls | entropy_cls)')
parser.add_argument('--save', type=bool, default=True)
