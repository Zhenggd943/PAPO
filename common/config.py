import argparse

parser = argparse.ArgumentParser(description='Global Setting')
parser.add_argument('--local', type=int, default=0, help='local or server')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--rootpath_local', type=str, default='E:\Experiment\CL\Continual_Learning\\', help='local save path')
parser.add_argument('--rootpath_server', type=str, default='E:\Experiment\CL\Continual_Learning\\', help='server save path')
global_args = parser.parse_args()

