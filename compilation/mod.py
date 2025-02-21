import pickle
from mmap import mmap
from multiprocessing.context import assert_spawning
import os, os.path as osp

import numpy as np

from convert import (
    build_quantized_mbv2
)


import tvm
from tvm import relay, te
from tvm.contrib import graph_executor

import warnings

def mod_save(
    mod,
    params=None,
    meta=None,
    path=".models/sample_net",
    mod_name="mod.ir",
    param_name="weights.params",
):
    os.makedirs(path, exist_ok=True)
    with open(osp.join(path, mod_name), "w") as fp:
        fp.write(str(mod))

    if params is not None:
        with open(osp.join(path, param_name), "wb") as fp:
            fp.write(tvm.runtime.save_param_dict(params))

    if meta is not None:
        with open(osp.join(path, mod_name.replace(".ir", ".pkl")), "wb") as fp:
            # fp.write(str(mod["main"]))
            pickle.dump(meta, fp)


def mod_load(
    path=".models/sample_net",
    mod_name="mod.ir",
    param_name="weights.params",
    meta=None,
    adapt_output_info=False,
):
    SEMVER = '#[version = "0.0.5"]\n'

    assert osp.exists(osp.join(path, mod_name)), "%s not found " % osp.join(
        path, mod_name
    )
    with open(osp.join(path, mod_name), "r") as fp:
        code = fp.read()
        if adapt_output_info:
            lines = code.split("\n")
            segs = lines[0].split("->")
            segs[-1] = "{"
            line1 = "".join(segs)
            lines[0] = line1
            code = "\n".join(lines)

    metatable = None
    if meta is not None:
        print("Loading meta table information")
        with open(osp.join(path, meta), "rb") as fp:
            metatable = pickle.load(fp)
            metatable = {
                "relay.Constant": [
                    relay.const(_, dtype=str(_.dtype)) for _ in metatable
                ]
            }

    mod_expr = tvm.parser.parse(
        SEMVER + code,
        "from_string",
        None,
        metatable,
    )
    # mod = tvm.IRModule.from_expr(mod_expr)
    mod = mod_expr
    mod = relay.transform.InferType()(mod)

    params = None
    if not osp.exists(osp.join(path, param_name)):
        warnings.warn("%s not exist! Load none params" % osp.join(path, param_name))
    else:
        with open(osp.join(path, param_name), "rb") as fp:
            bin_params = fp.read()
        params = dict(tvm.runtime.load_param_dict(bin_params))
    return mod, params


class MRun:
    def __init__(self, mod=None, mpath=None, weights=None, wpath=None, target="llvm"):
        assert not mod or not mpath
        assert mod or mpath
        self.dev = tvm.cuda()
        if mod:
            self.mod = mod
        elif mpath:
            with open(mpath, "r") as fp:
                code = fp.read()
            SEMVER = '#[version = "0.0.5"]\n'
            mod_expr = tvm.parser.parse_expr(SEMVER + code)
            mod = tvm.IRModule.from_expr(mod_expr)
            mod = relay.transform.InferType()(mod)
            self.mod = mod

        self.vs = relay.analysis.all_vars(mod["main"])
        self.lib = relay.build(mod, target=tvm.target.cuda(arch="sm_75"))
        self.g = graph_executor.GraphModule(self.lib["default"](tvm.cuda()))

        if wpath:
            print(f"weights loaded from {wpath}")
            with open(wpath, "rb") as fp:
                bin_params = fp.read()
            params = dict(tvm.runtime.load_param_dict(bin_params))
            new_params = {}
            for k, v in params.items():
                if k[0].isdigit():
                    k = "v" + k
                new_params[k] = v
            self.bind_data(new_params)
            self.new_params = new_params
        elif weights:
            self.bind_data(weights)
            self.new_params = weights

    def randomly_init_weights(self, loc=0, scale=1):
        tp = {}
        for idx, v in enumerate(self.vs):
            shape = [int(_) for _ in v.type_annotation.shape]
            dtype = str(v.type_annotation.dtype)
            # print(v.type_annotation.shape, v.type_annotation.dtype)
            p = np.ones(shape).astype(str(dtype))
            p = np.random.normal(loc=loc, scale=scale, size=shape).astype(str(dtype))
            tp[str(v.name_hint)] = p
        self.bind_data(tp)
        return tp

    def bind_data(self, data):
        if isinstance(data, np.ndarray):
            self.g.set_input(self.data_names[0], data)
        elif isinstance(data, dict):
            for k, v in data.items():
                try:
                    self.g.set_input(k, v)
                except (tvm._ffi.base.TVMError, ValueError):
                    t = self.g.get_input(k)
                    print(
                        f"Failed to set_input for |{k}|, feed-in: {v.shape, v.dtype}, expected {t.shape, t.dtype}\n"
                    )
                    # raise
                    exit(0)

    def __call__(self, data):
        self.bind_data(data)
        self.g.run()
        r = []
        for idx in range(self.g.get_num_outputs()):
            _r = self.g.get_output(idx)
            r.append(_r)
        return r


class ComputeDAG:
    def __init__(
        self,
        path,
        mod_name="mod.ir",
        param_name="weights.params",
        target="llvm",
        dev=tvm.cuda(0),
    ):
        self.path = path

        self.target = target
        self.dev = dev
        self.mod2lib = dict()
        self.lib2mod = dict()
        self.total_args = []

        mod, params = mod_load(path, mod_name, param_name)
        self.mod = mod
        if params is None:
            params = {}
        self.mod_params = params
        # print(param_name, self.mod_params.keys())
        # exit(0)

    def compile(self, mod_override=None, optimize=False):
        # with tvm.transform.PassContext(opt_level = opt_level):
        if mod_override is None:
            mod_to_build = self.mod
        else:
            mod_to_build = mod_override
        if optimize:
            mod_to_build = tvm.transform.Sequential(
                [
                    relay.transform.DeadCodeElimination(),
                    relay.transform.ToGraphNormalForm(),
                    relay.transform.FoldConstant(),
                    relay.transform.SimplifyExpr(),
                ]
            )(mod_to_build)

        lib = relay.build(mod_to_build, target=tvm.target.cuda(arch="sm_75"), params=self.mod_params)
        lib_params = lib.get_params()

        vs = relay.analysis.all_vars(self.mod["main"])
        # the first elem is the input
        # self.input_name = vs[0].name_hint
        vname = [v.name_hint for v in vs][1:]
        func_args = []
        data_args = []
        total_args = []
        for arg in relay.analysis.all_vars(self.mod["main"]):
            vname = arg.name_hint
            if vname.startswith("x"):
                # TODO: this is a dirty fix to "let" assignments in TVMIR
                # TODO: find the proper binding in TVM underlying calls.
                continue
            if vname in self.mod_params.keys() or vname[1:] in self.mod_params.keys():
                # TODO: dirty fix to variable likes v0.weight
                func_args.append(arg)
            else:
                data_args.append(arg)
            # print("==" * 40)
            # print(vname, self.mod_params.keys() , func_args, data_args, sep="\n")
            total_args.append(arg)

        self.total_args = total_args
        self.data_args = data_args
        self.func_args = func_args
        print(f"data_args: @{len(data_args)}", [_.name_hint for _ in data_args])
        print(f"func_args: @{len(func_args)}", [_.name_hint for _ in func_args])

        # check vars and matched shape
        assert len(func_args) <= len(
            lib_params.keys()
        ), f"{len(func_args)}|{len(lib_params.keys())}\n{func_args}\n{lib_params.keys()}"

        for idx, args in enumerate(func_args):
            v = args.name_hint
            p1 = self.mod_params[v]
            p2 = lib_params["p" + str(idx)]
            # print(p1.shape, p2.shape, p1.shape ==  p2.shape)
            assert (
                p1.shape == p2.shape
            ), f"Shape mismatch for |{v}|, expected: {p1.shape}, get {p2.shape}"
            self.mod2lib[v] = "p" + str(idx)
            self.lib2mod["p" + str(idx)] = v

        self.data_names = [_.name_hint for _ in data_args]
        self.lib = lib
        self.lib_params = lib.get_params()
        self.g = graph_executor.GraphModule(lib["default"](self.dev))

    def bind_data(self, data):
        if isinstance(data, np.ndarray):
            self.g.set_input(self.data_names[0], data)
        elif isinstance(data, dict):
            for k, v in data.items():
                assert k in self.data_names
                self.g.set_input(k, v)

    def __call__(self, data):
        self.bind_data(data)
        self.g.run()
        r = []
        for idx in range(self.g.get_num_outputs()):
            _r = self.g.get_output(idx)
            r.append(_r)
        return r

    def get_params(self):
        return self.mod_params

    def set_params(self, new_param: dict):
        new_lib_params = dict()
        for k, v in new_param.items():
            assert k in self.mod_params, f"[{k}] is unseen is previous parameters."
            new_k = self.mod2lib[k]
            new_v = tvm.nd.array(v, self.dev)
            new_lib_params[new_k] = new_v
            self.mod_params[k] = new_v
        self.g.load_params(tvm.runtime.save_param_dict(new_lib_params))

    def load(self, path):
        warnings.warn("DAG.load function is deprecated!")
        mod, params = mod_load(path)
        self.mod = mod
        self.mod_params = params

    def save(self, path):
        warnings.warn("DAG.save function is deprecated!")
        mod_save(self.mod, self.mod_params, self.path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bias_update', type=int, default=51,
                      help='Number of bias layer to update (default: 0)')
    parser.add_argument('--weight_idx', type=int, default=0,
                      help='Layer index to update (default: 0)')
    parser.add_argument('--num_classes', type=int, default=10,
                      help='Number of classes (default: 0)')
    parser.add_argument('--bias_only', action=argparse.BooleanOptionalAction,
                      help='Number of classes (default: 0)')

    args = parser.parse_args()

    from convert import build_quantized_mcunet
    import os, sys
    import torch

    package_path = os.path.abspath("/home/andrealavi/tirocinio/tiny-training/compilation")  # Update this to the actual path
    sys.path.append(package_path)

    from convert.pth2ir import pth_model_to_ir

    num_classes = 10
    rs = 128

    model, _ = build_quantized_mbv2(num_classes=num_classes)

    fwd_mod, real_params, scale_params, op_idx = pth_model_to_ir(model, input_res=[1, 3, rs, rs], num_classes=num_classes)

    cfg_idx =  {
        'enable_backward_config': 1, 'n_bias_update': args.n_bias_update, 'weight_update_ratio': [1], 'manual_weight_idx': [args.weight_idx], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0,
    }

    cfg_bias_only = {
        'enable_backward_config': 1, 'n_bias_update': args.n_bias_update, 'n_weight_update': 0, 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0,
    }

    cfg = {}

    from convert import generated_backward_graph

    if args.bias_only:
        cfg = cfg_bias_only
    else:
        cfg = cfg_idx

    bwd_mod, bwd_names, sparse_meta_info = generated_backward_graph(fwd_mod, op_idx, method="sparse_bp", sparse_bp_config=cfg, int8_bp=False)


    package_path = os.path.abspath("/home/andrealavi/tirocinio/tiny-training/algorithm")  # Update this to the actual path
    sys.path.append(package_path)

    from algorithm.core.utils import dist

    from algorithm.core.dataset.dataset_entry import build_dataset
    from algorithm.core.utils.config import load_config_from_file, configs

    load_config_from_file("../configs/transfer.yaml")

    dataset = build_dataset()
    data_loader = dict()

    for split in dataset:
        sampler = torch.utils.data.DistributedSampler( # Sampler is used to take random samples from the dataset
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=configs.manual_seed,
            shuffle=(split == 'train')) # Shuffles only if split is true

        data_loader[split] = torch.utils.data.DataLoader( # Loads data based on sampler
            dataset[split],
            batch_size=configs.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=configs.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    from time import time

    comp_start = time()
    comp_mod = MRun(bwd_mod, target="cuda")
    comp_end = time()

    comp_time = comp_end - comp_start

    print("modello compilato")

    comp_mod.randomly_init_weights()


    for _, (images, labels) in enumerate(data_loader["val"]):
        img = images[0].cpu()
        label = labels[0].cpu()

        img = torch.reshape(img, shape=(1,3,128,128))

        data = {
            "input": img.numpy().astype(np.int8),

            # v0 section
            "v0_weight": model[0].weight.cpu().detach().numpy().astype(np.int8),
            "v0_bias": model[0].bias.cpu().detach().numpy().astype(np.int32),
            "v0_zero_x": model[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v0_zero_y": model[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v0_scale": model[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v1 section
            "v1_conv_0_weight": model[1][0].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v1_conv_0_bias": model[1][0].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v1_conv_0_zero_x": model[1][0].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v1_conv_0_zero_y": model[1][0].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v1_conv_0_scale": model[1][0].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v1_conv_1_weight": model[1][0].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v1_conv_1_bias": model[1][0].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v1_conv_1_zero_x": model[1][0].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v1_conv_1_zero_y": model[1][0].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v1_conv_1_scale": model[1][0].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v2 section
            "v2_conv_0_weight": model[1][1].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v2_conv_0_bias": model[1][1].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v2_conv_0_zero_x": model[1][1].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v2_conv_0_zero_y": model[1][1].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v2_conv_0_scale": model[1][1].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v2_conv_1_weight": model[1][1].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v2_conv_1_bias": model[1][1].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v2_conv_1_zero_x": model[1][1].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v2_conv_1_zero_y": model[1][1].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v2_conv_1_scale": model[1][1].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v2_conv_2_weight": model[1][1].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v2_conv_2_bias": model[1][1].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v2_conv_2_zero_x": model[1][1].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v2_conv_2_zero_y": model[1][1].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v2_conv_2_scale": model[1][1].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v3 section (with qadd)
            "v3_conv_0_weight": model[1][2].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v3_conv_0_bias": model[1][2].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v3_conv_0_zero_x": model[1][2].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v3_conv_0_zero_y": model[1][2].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v3_conv_0_scale": model[1][2].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v3_conv_1_weight": model[1][2].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v3_conv_1_bias": model[1][2].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v3_conv_1_zero_x": model[1][2].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v3_conv_1_zero_y": model[1][2].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v3_conv_1_scale": model[1][2].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v3_conv_2_weight": model[1][2].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v3_conv_2_bias": model[1][2].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v3_conv_2_zero_x": model[1][2].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v3_conv_2_zero_y": model[1][2].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v3_conv_2_scale": model[1][2].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v4 section
            "v4_conv_0_weight": model[1][3].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v4_conv_0_bias": model[1][3].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v4_conv_0_zero_x": model[1][3].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v4_conv_0_zero_y": model[1][3].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v4_conv_0_scale": model[1][3].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v4_conv_1_weight": model[1][3].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v4_conv_1_bias": model[1][3].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v4_conv_1_zero_x": model[1][3].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v4_conv_1_zero_y": model[1][3].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v4_conv_1_scale": model[1][3].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v4_conv_2_weight": model[1][3].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v4_conv_2_bias": model[1][3].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v4_conv_2_zero_x": model[1][3].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v4_conv_2_zero_y": model[1][3].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v4_conv_2_scale": model[1][3].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v5 section (with qadd)
            "v5_conv_0_weight": model[1][4].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v5_conv_0_bias": model[1][4].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v5_conv_0_zero_x": model[1][4].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v5_conv_0_zero_y": model[1][4].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v5_conv_0_scale": model[1][4].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v5_conv_1_weight": model[1][4].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v5_conv_1_bias": model[1][4].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v5_conv_1_zero_x": model[1][4].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v5_conv_1_zero_y": model[1][4].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v5_conv_1_scale": model[1][4].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v5_conv_2_weight": model[1][4].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v5_conv_2_bias": model[1][4].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v5_conv_2_zero_x": model[1][4].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v5_conv_2_zero_y": model[1][4].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v5_conv_2_scale": model[1][4].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v6 section
            "v6_conv_0_weight": model[1][5].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v6_conv_0_bias": model[1][5].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v6_conv_0_zero_x": model[1][5].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v6_conv_0_zero_y": model[1][5].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v6_conv_0_scale": model[1][5].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v6_conv_1_weight": model[1][5].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v6_conv_1_bias": model[1][5].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v6_conv_1_zero_x": model[1][5].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v6_conv_1_zero_y": model[1][5].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v6_conv_1_scale": model[1][5].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v6_conv_2_weight": model[1][5].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v6_conv_2_bias": model[1][5].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v6_conv_2_zero_x": model[1][5].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v6_conv_2_zero_y": model[1][5].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v6_conv_2_scale": model[1][5].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v7 section (with qadd)
            "v7_conv_0_weight": model[1][6].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v7_conv_0_bias": model[1][6].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v7_conv_0_zero_x": model[1][6].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v7_conv_0_zero_y": model[1][6].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v7_conv_0_scale": model[1][6].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v7_conv_1_weight": model[1][6].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v7_conv_1_bias": model[1][6].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v7_conv_1_zero_x": model[1][6].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v7_conv_1_zero_y": model[1][6].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v7_conv_1_scale": model[1][6].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v7_conv_2_weight": model[1][6].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v7_conv_2_bias": model[1][6].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v7_conv_2_zero_x": model[1][6].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v7_conv_2_zero_y": model[1][6].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v7_conv_2_scale": model[1][6].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v8 section
            "v8_conv_0_weight": model[1][7].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v8_conv_0_bias": model[1][7].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v8_conv_0_zero_x": model[1][7].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v8_conv_0_zero_y": model[1][7].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v8_conv_0_scale": model[1][7].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v8_conv_1_weight": model[1][7].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v8_conv_1_bias": model[1][7].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v8_conv_1_zero_x": model[1][7].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v8_conv_1_zero_y": model[1][7].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v8_conv_1_scale": model[1][7].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v8_conv_2_weight": model[1][7].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v8_conv_2_bias": model[1][7].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v8_conv_2_zero_x": model[1][7].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v8_conv_2_zero_y": model[1][7].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v8_conv_2_scale": model[1][7].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v9 section (with qadd)
            "v9_conv_0_weight": model[1][8].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v9_conv_0_bias": model[1][8].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v9_conv_0_zero_x": model[1][8].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v9_conv_0_zero_y": model[1][8].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v9_conv_0_scale": model[1][8].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v9_conv_1_weight": model[1][8].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v9_conv_1_bias": model[1][8].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v9_conv_1_zero_x": model[1][8].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v9_conv_1_zero_y": model[1][8].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v9_conv_1_scale": model[1][8].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v9_conv_2_weight": model[1][8].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v9_conv_2_bias": model[1][8].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v9_conv_2_zero_x": model[1][8].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v9_conv_2_zero_y": model[1][8].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v9_conv_2_scale": model[1][8].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v10 section (with qadd)
            "v10_conv_0_weight": model[1][9].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v10_conv_0_bias": model[1][9].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v10_conv_0_zero_x": model[1][9].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v10_conv_0_zero_y": model[1][9].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v10_conv_0_scale": model[1][9].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v10_conv_1_weight": model[1][9].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v10_conv_1_bias": model[1][9].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v10_conv_1_zero_x": model[1][9].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v10_conv_1_zero_y": model[1][9].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v10_conv_1_scale": model[1][9].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v10_conv_2_weight": model[1][9].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v10_conv_2_bias": model[1][9].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v10_conv_2_zero_x": model[1][9].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v10_conv_2_zero_y": model[1][9].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v10_conv_2_scale": model[1][9].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v11 section
            "v11_conv_0_weight": model[1][10].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v11_conv_0_bias": model[1][10].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v11_conv_0_zero_x": model[1][10].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v11_conv_0_zero_y": model[1][10].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v11_conv_0_scale": model[1][10].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v11_conv_1_weight": model[1][10].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v11_conv_1_bias": model[1][10].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v11_conv_1_zero_x": model[1][10].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v11_conv_1_zero_y": model[1][10].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v11_conv_1_scale": model[1][10].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v11_conv_2_weight": model[1][10].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v11_conv_2_bias": model[1][10].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v11_conv_2_zero_x": model[1][10].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v11_conv_2_zero_y": model[1][10].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v11_conv_2_scale": model[1][10].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v12 section (with qadd)
            "v12_conv_0_weight": model[1][11].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v12_conv_0_bias": model[1][11].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v12_conv_0_zero_x": model[1][11].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v12_conv_0_zero_y": model[1][11].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v12_conv_0_scale": model[1][11].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v12_conv_1_weight": model[1][11].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v12_conv_1_bias": model[1][11].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v12_conv_1_zero_x": model[1][11].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v12_conv_1_zero_y": model[1][11].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v12_conv_1_scale": model[1][11].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v12_conv_2_weight": model[1][11].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v12_conv_2_bias": model[1][11].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v12_conv_2_zero_x": model[1][11].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v12_conv_2_zero_y": model[1][11].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v12_conv_2_scale": model[1][11].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v13 section (with qadd)
            "v13_conv_0_weight": model[1][12].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v13_conv_0_bias": model[1][12].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v13_conv_0_zero_x": model[1][12].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v13_conv_0_zero_y": model[1][12].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v13_conv_0_scale": model[1][12].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v13_conv_1_weight": model[1][12].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v13_conv_1_bias": model[1][12].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v13_conv_1_zero_x": model[1][12].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v13_conv_1_zero_y": model[1][12].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v13_conv_1_scale": model[1][12].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v13_conv_2_weight": model[1][12].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v13_conv_2_bias": model[1][12].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v13_conv_2_zero_x": model[1][12].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v13_conv_2_zero_y": model[1][12].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v13_conv_2_scale": model[1][12].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v14 section (with qadd)
            "v14_conv_0_weight": model[1][13].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v14_conv_0_bias": model[1][13].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v14_conv_0_zero_x": model[1][13].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v14_conv_0_zero_y": model[1][13].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v14_conv_0_scale": model[1][13].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v14_conv_1_weight": model[1][13].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v14_conv_1_bias": model[1][13].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v14_conv_1_zero_x": model[1][13].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v14_conv_1_zero_y": model[1][13].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v14_conv_1_scale": model[1][13].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v14_conv_2_weight": model[1][13].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v14_conv_2_bias": model[1][13].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v14_conv_2_zero_x": model[1][13].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v14_conv_2_zero_y": model[1][13].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v14_conv_2_scale": model[1][13].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v15 section (with qadd)
            "v15_conv_0_weight": model[1][14].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v15_conv_0_bias": model[1][14].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v15_conv_0_zero_x": model[1][14].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v15_conv_0_zero_y": model[1][14].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v15_conv_0_scale": model[1][14].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v15_conv_1_weight": model[1][14].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v15_conv_1_bias": model[1][14].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v15_conv_1_zero_x": model[1][14].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v15_conv_1_zero_y": model[1][14].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v15_conv_1_scale": model[1][14].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v15_conv_2_weight": model[1][14].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v15_conv_2_bias": model[1][14].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v15_conv_2_zero_x": model[1][14].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v15_conv_2_zero_y": model[1][14].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v15_conv_2_scale": model[1][14].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v16 section (with qadd)
            "v16_conv_0_weight": model[1][15].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v16_conv_0_bias": model[1][15].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v16_conv_0_zero_x": model[1][15].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v16_conv_0_zero_y": model[1][15].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v16_conv_0_scale": model[1][15].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v16_conv_1_weight": model[1][15].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v16_conv_1_bias": model[1][15].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v16_conv_1_zero_x": model[1][15].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v16_conv_1_zero_y": model[1][15].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v16_conv_1_scale": model[1][15].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v16_conv_2_weight": model[1][15].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v16_conv_2_bias": model[1][15].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v16_conv_2_zero_x": model[1][15].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v16_conv_2_zero_y": model[1][15].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v16_conv_2_scale": model[1][15].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32),

            # v17 section (with qadd)
            "v17_conv_0_weight": model[1][16].conv[0].weight.cpu().detach().numpy().astype(np.int8),
            "v17_conv_0_bias": model[1][16].conv[0].bias.cpu().detach().numpy().astype(np.int32),
            "v17_conv_0_zero_x": model[1][16].conv[0].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v17_conv_0_zero_y": model[1][16].conv[0].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v17_conv_0_scale": model[1][16].conv[0].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v17_conv_1_weight": model[1][16].conv[1].weight.cpu().detach().numpy().astype(np.int8),
            "v17_conv_1_bias": model[1][16].conv[1].bias.cpu().detach().numpy().astype(np.int32),
            "v17_conv_1_zero_x": model[1][16].conv[1].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v17_conv_1_zero_y": model[1][16].conv[1].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v17_conv_1_scale": model[1][16].conv[1].get_buffer("effective_scale").cpu().numpy().astype(np.float32),
            "v17_conv_2_weight": model[1][16].conv[2].weight.cpu().detach().numpy().astype(np.int8),
            "v17_conv_2_bias": model[1][16].conv[2].bias.cpu().detach().numpy().astype(np.int32),
            "v17_conv_2_zero_x": model[1][16].conv[2].get_buffer("zero_x").cpu().numpy().astype(np.int8).reshape(1),
            "v17_conv_2_zero_y": model[1][16].conv[2].get_buffer("zero_y").cpu().numpy().astype(np.int8).reshape(1),
            "v17_conv_2_scale": model[1][16].conv[2].get_buffer("effective_scale").cpu().numpy().astype(np.float32)
        }

        out_start = time()
        out = comp_mod(data)
        out_end = time()

        out_time = out_end - out_start

        print("output calcolato")

        with open(f"./tests/b_{args.n_bias_update}_w_{args.weight_idx}_time", "w") as f:
            f.write(f"Updated bias: {args.n_bias_update}\n")
            f.write(f"Weight Layer: {args.weight_idx}\n")
            f.write(f"compilation time: {comp_time}\n")
            f.write(f"execution time: {out_time}\n")

        break
