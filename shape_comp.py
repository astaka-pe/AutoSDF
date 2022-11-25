import argparse
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from pytorch3d.io import IO

cudnn.benchmark = True

import utils
from utils.util_3d import init_mesh_renderer, sdf_to_mesh
from utils.qual_util import save_mesh_as_gif
from utils.demo_util import get_shape_comp_model, get_shape_comp_opt
from utils.qual_util import get_partial_shape_by_range
from preprocess.process_one_mesh import process_obj


def get_parser():
    parser = argparse.ArgumentParser(description="MeshConv")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="output.obj")
    parser.add_argument("-model_path", type=str, default="saved_ckpt/rand_tf-snet_code-all-LR1e-4-clean-epoch200.pth")
    args = parser.parse_args()
    return args

class AutoSDFComp():
    def __init__(self, mesh_path, model_path):
        self.load_pretrained_model(model_path)
        self.load_mesh(mesh_path)

    def load_pretrained_model(self, path):
        self.opt = get_shape_comp_opt(gpu_id=0)
        self.model = get_shape_comp_model(self.opt)
        self.model.load_ckpt(path)
        self.model.eval()

    def load_mesh(self, path):
        sdf_file = process_obj(path)
        min_x, max_x = -1., 1.
        min_y, max_y = -1., 0.99
        min_z, max_z = -1., 1.
        sdf = utils.util_3d.read_sdf(sdf_file).to(self.opt.device)
        input_range = {'x1': min_x, 'x2': max_x, 'y1': min_y, 'y2': max_y, 'z1': min_z, 'z2': max_z}
        self.input = get_partial_shape_by_range(sdf, input_range)

    def completion(self):
        _, comp_sdf = self.model.shape_comp(self.input, bs=9, topk=30)
        gen_mesh = sdf_to_mesh(comp_sdf)
        return gen_mesh

def save_mesh(path, mesh):
    IO().save_mesh(mesh, path)


def main():
    args = get_parser()
    compnet = AutoSDFComp(args.input, args.model_path)
    out_mesh = compnet.completion()
    for i, m in enumerate(out_mesh):
        save_mesh("{}_{}".format(i, args.output), m)

if __name__ == "__main__":
    main()