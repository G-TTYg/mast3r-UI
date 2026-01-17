#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# A new Gradio demo for MASt3R with enhanced UI and features
# --------------------------------------------------------
import gradio as gr
import torch
import os
import tempfile
from contextlib import nullcontext
import functools
import math
import numpy as np
import trimesh
import copy
from scipy.spatial.transform import Rotation
import shutil
import argparse
import builtins
import datetime

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs

try:
    from mast3r.retrieval.processor import Retriever
    has_retrieval = True
except Exception:
    has_retrieval = False

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes


import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True

# --- Localization Dictionary ---
i18n = {
    "en": {
        "title": "MASt3R Enhanced UI",
        "upload_files": "Upload Image Files",
        "run": "Run Reconstruction",
        # New Configuration section
        "config_title": "Configuration",
        "device": "Device",
        "model": "Model",
        "custom_model_path": "Custom Model Path",
        "custom_model_path_placeholder": "Enter path to your custom .pth file",
        "retrieval_model": "Retrieval Model",
        "retrieval_model_none": "None",
        "custom_retrieval_model_path": "Custom Retrieval Model Path",
        "custom_retrieval_model_path_placeholder": "Enter path to your custom retrieval .pth file",

        # Parameters
        "optimization_params": "Optimization Parameters",
        "coarse_lr": "Coarse LR",
        "coarse_lr_info": "Learning rate for the coarse alignment phase.",
        "coarse_iter": "Coarse Iterations",
        "coarse_iter_info": "Number of iterations for coarse alignment.",
        "fine_lr": "Fine LR",
        "fine_lr_info": "Learning rate for the refinement phase.",
        "fine_iter": "Fine Iterations",
        "fine_iter_info": "Number of iterations for refinement.",
        "optim_level": "Optimization Level",
        "optim_level_info": "Controls the depth of the optimization process.",
        "matching_conf_thr": "Matching Confidence Threshold",
        "matching_conf_thr_info": "Confidence threshold for matching points before falling back to 3D regression.",
        "shared_intrinsics": "Shared Intrinsics",
        "shared_intrinsics_info": "Assume all cameras share the same intrinsic parameters. Useful for videos shot with a single camera.",

        "scenegraph_params": "Scene Graph Parameters",
        "scenegraph_type": "Scene Graph Type",
        "scenegraph_type_info": "Defines how image pairs are selected for matching.",
        "scenegraph_window_size": "Window Size",
        "cyclic_sequence": "Cyclic Sequence",
        "reference_id": "Reference ID",
        "retrieval_key_images": "Retrieval: Num. key images",
        "retrieval_neighbors": "Retrieval: Num neighbors",
        "sg_complete": "complete: all possible image pairs",
        "sg_retrieval": "retrieval: connect views based on similarity",
        "sg_swin": "swin: sliding window",
        "sg_logwin": "logwin: sliding window with long range",
        "sg_oneref": "oneref: match one image with all",


        "visualization_params": "Visualization Parameters",
        "min_conf_thr": "Min. Confidence Threshold",
        "min_conf_thr_info": "Minimum confidence for a 3D point to be visualized.",
        "cam_size": "Camera Size",
        "cam_size_info": "Controls the size of the camera frustums in the 3D view.",
        "tsdf_threshold": "TSDF Threshold",
        "tsdf_threshold_info": "Threshold for TSDF-based surface reconstruction. 0 disables it.",
        "as_pointcloud": "Display as Point Cloud",
        "mask_sky": "Mask Sky",
        "clean_depth": "Clean-up Depth Maps",
        "transparent_cams": "Transparent Cameras",
        "uploaded_files": "Uploaded Files",
    },
    "zh": {
        "title": "MASt3R 增强版UI",
        "upload_files": "上传图像文件",
        "uploaded_files": "已上传文件",
        "run": "开始重建",
        # New Configuration section
        "config_title": "配置",
        "device": "设备",
        "model": "模型",
        "custom_model_path": "自定义模型路径",
        "custom_model_path_placeholder": "请输入您的自定义 .pth 文件路径",
        "retrieval_model": "检索模型",
        "retrieval_model_none": "无",
        "custom_retrieval_model_path": "自定义检索模型路径",
        "custom_retrieval_model_path_placeholder": "请输入您的自定义检索 .pth 文件路径",

        # Parameters
        "optimization_params": "优化参数",
        "coarse_lr": "粗略对齐学习率 (Coarse LR)",
        "coarse_lr_info": "用于粗略对齐阶段的学习率。",
        "coarse_iter": "粗略对齐迭代次数",
        "coarse_iter_info": "粗略对齐阶段的迭代次数。",
        "fine_lr": "精细对齐学习率 (Fine LR)",
        "fine_lr_info": "用于精细调整阶段的学习率。",
        "fine_iter": "精细对齐迭代次数",
        "fine_iter_info": "精细调整阶段的迭代次数。",
        "optim_level": "优化级别",
        "optim_level_info": "控制优化的深度和范围。",
        "matching_conf_thr": "匹配置信度阈值",
        "matching_conf_thr_info": "在回退到3D回归前，用于点匹配的置信度阈值。",
        "shared_intrinsics": "共享内参",
        "shared_intrinsics_info": "假设所有相机共享相同的内参。适用于使用单个相机拍摄的视频。",

        "scenegraph_params": "场景图参数",
        "scenegraph_type": "场景图类型",
        "scenegraph_type_info": "定义如何选择图像对进行匹配。",
        "scenegraph_window_size": "窗口大小",
        "cyclic_sequence": "循环序列",
        "reference_id": "参考图像ID",
        "retrieval_key_images": "检索: 关键图像数",
        "retrieval_neighbors": "检索: 邻居数",
        "sg_complete": "完整连接: 所有可能的图像对",
        "sg_retrieval": "检索: 基于相似度连接视图",
        "sg_swin": "滑动窗口",
        "sg_logwin": "对数滑动窗口: 带远距离连接",
        "sg_oneref": "单一参考: 将一张图像与所有其他图像匹配",


        "visualization_params": "可视化参数",
        "min_conf_thr": "最小置信度阈值",
        "min_conf_thr_info": "一个3D点要被可视化所需的最小置信度。",
        "cam_size": "相机大小",
        "cam_size_info": "控制3D视图中相机视锥的大小。",
        "tsdf_threshold": "TSDF 阈值",
        "tsdf_threshold_info": "用于基于TSDF的表面重建的阈值。设置为0则禁用。",
        "as_pointcloud": "以点云形式显示",
        "mask_sky": "遮蔽天空",
        "clean_depth": "清理深度图",
        "transparent_cams": "透明相机",
    }
}

class SparseGAState:
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False)
    parser_url.add_argument("--server_name", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--silent", action='store_true', default=False)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    parser.prog = 'mast3r_app'
    return parser

def _convert_scene_output(outdir, export_format, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                          cam_color=None, as_pointcloud=False,
                          transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    outfile = os.path.join(outdir, f'scene{export_format}')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0, export_format=".glb"):
    if scene_state is None:
        return None

    outdir = scene_state.cache_dir
    if outdir is None:
        return None

    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output(outdir, export_format, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                 transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def get_reconstructed_scene(outdir, gradio_delete_cache, model, retrieval_model, device, silent, image_size,
                            current_scene_state, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr,
                            matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, export_format=".glb", **kw):
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))
        scene_graph_params.append(str(refid))

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert has_retrieval
        assert retrieval_model is not None
        with torch.no_grad():
            sim_matrix = retrieval_model(filelist)
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0

    if current_scene_state is not None and not current_scene_state.should_delete and current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)

    if current_scene_state is not None and not current_scene_state.should_delete and current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix=f'_scene{export_format}', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh, export_format=export_format)
    return scene_state, outfile

def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize, min_winsize = 1, 1
    win_cyclic_old_val = win_cyclic

    winsize = gr.Slider(visible=False)
    win_cyclic = gr.Checkbox(visible=False)
    graph_opt = gr.Column(visible=False)
    refid = gr.Slider(visible=False)

    if scenegraph_type in ["swin", "logwin"]:
        if scenegraph_type == "swin":
            if win_cyclic_old_val:
                max_winsize = max(1, math.ceil((num_files - 1) / 2))
            else:
                max_winsize = num_files - 1
        else:
            if win_cyclic_old_val:
                half_size = math.ceil((num_files - 1) / 2)
                max_winsize = max(1, math.ceil(math.log(half_size, 2)))
            else:
                max_winsize = max(1, math.ceil(math.log(num_files, 2)))

        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=min_winsize, maximum=max_winsize, step=1, visible=True)
        win_cyclic = gr.Checkbox(value=win_cyclic_old_val, label="Cyclic sequence", visible=True)
        graph_opt = gr.Column(visible=True)
        refid = gr.Slider(visible=False)

    elif scenegraph_type == "retrieval":
        graph_opt = gr.Column(visible=True)
        winsize = gr.Slider(label="Retrieval: Num. key images", value=min(20, num_files),
                                minimum=0, maximum=num_files, step=1, visible=True)
        win_cyclic = gr.Checkbox(visible=False)
        refid = gr.Slider(label="Retrieval: Num neighbors", value=min(num_files - 1, 10), minimum=1,
                              maximum=num_files - 1, step=1, visible=True)

    elif scenegraph_type == "oneref":
        graph_opt = gr.Column(visible=True)
        winsize = gr.Slider(visible=False)
        win_cyclic = gr.Checkbox(visible=False)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)

    return graph_opt, winsize, win_cyclic, refid

def get_text(lang, key):
    return i18n[lang].get(key, key)

def main(args):
    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    # Global cache for the model
    model_cache = {
        "model": None,
        "weights_path": None,
        "device": None,
        "retrieval_model": None,
        "retrieval_weights_path": None,
    }

    def get_retrieval_model(retrieval_model_name, custom_retrieval_model_path, main_model, device):
        if retrieval_model_name == get_text(lang_state.value, "retrieval_model_none"):
            return None

        if retrieval_model_name == "custom":
            weights_path = custom_retrieval_model_path
        else:
            weights_path = "naver/" + retrieval_model_name

        if (model_cache["retrieval_model"] is not None and
            model_cache["retrieval_weights_path"] == weights_path and
            model_cache["device"] == device):
            return model_cache["retrieval_model"]

        retrieval_model = Retriever(weights_path, backbone=main_model, device=device)

        model_cache["retrieval_model"] = retrieval_model
        model_cache["retrieval_weights_path"] = weights_path
        model_cache["device"] = device

        return retrieval_model

    def get_model(model_name, custom_model_path, device):
        if model_name == "custom":
            weights_path = custom_model_path
        else:
            weights_path = "naver/" + model_name

        if (model_cache["model"] is not None and
            model_cache["weights_path"] == weights_path and
            model_cache["device"] == device):
            return model_cache["model"]

        model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)

        model_cache["model"] = model
        model_cache["weights_path"] = weights_path
        model_cache["device"] = device

        return model

    def run_reconstruction(scene_state, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                           as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                           scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics,
                           model_name_dd, custom_model_path_tb, device_r, export_format_dd,
                           retrieval_model_name_dd, custom_retrieval_model_path_tb,
                           progress=gr.Progress()):

        progress(0, desc="Loading model...")
        model = get_model(model_name_dd, custom_model_path_tb, device_r)
        retrieval_model = get_retrieval_model(retrieval_model_name_dd, custom_retrieval_model_path_tb, model, device_r)

        chkpt_tag = hash_md5(model_cache["weights_path"])

        cache_path = os.path.join(args.tmp_dir, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)

        recon_fun = functools.partial(get_reconstructed_scene, cache_path, args.gradio_delete_cache, model,
                                        retrieval_model, device_r, args.silent, args.image_size)

        progress(0.1, desc="Running reconstruction...")
        scene_state, outmodel = recon_fun(scene_state, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                          as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                          scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, export_format=export_format_dd)
        progress(1.0, desc="Done!")
        return scene_state, outmodel

    with tempfile.TemporaryDirectory(suffix='_mast3r_gradio_persistent_cache') as tmpdirname:
        args.tmp_dir = tmpdirname

        model_from_scene_fun = functools.partial(get_3D_model_from_scene, args.silent)

        css = """
        .gradio-container {margin: 0 !important; min-width: 100%}
        .image-upload .h-full {
            max-height: 200px;
            overflow-y: auto;
        }
        """
        with gr.Blocks(css=css, title="MASt3R Enhanced UI") as demo:
            scene_state = gr.State(None)
            lang_state = gr.State("en")
            retrieval_model_state = gr.State(get_text("en", "retrieval_model_none"))

            with gr.Row():
                title_html = gr.HTML('<h2 id="title" style="text-align: left; flex-grow: 1; margin: 0;">MASt3R Enhanced UI</h2>')
                lang_radio = gr.Radio(["English", "中文"], value="English", label="Language", show_label=False, container=False, scale=0)

            with gr.Accordion("Configuration", open=True) as config_accordion:
                with gr.Row():
                    device = gr.Radio(["cpu", "cuda"], value="cpu" if not torch.cuda.is_available() else "cuda", label="Device")
                    model_name = gr.Dropdown(["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", "custom"], label="Model")
                    custom_model_path = gr.Textbox(label="Custom Model Path", placeholder="Enter path to your custom .pth file", visible=False)

                def toggle_custom_model_path(model_name):
                    return gr.update(visible=model_name == "custom")

                model_name.change(toggle_custom_model_path, inputs=model_name, outputs=custom_model_path)

                retrieval_model_name = gr.Dropdown(
                    [get_text("en", "retrieval_model_none"),
                     "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree",
                     "custom"],
                    label=get_text("en", "retrieval_model"),
                    value=get_text("en", "retrieval_model_none")
                )
                custom_retrieval_model_path = gr.Textbox(
                    label=get_text("en", "custom_retrieval_model_path"),
                    placeholder=get_text("en", "custom_retrieval_model_path_placeholder"),
                    visible=False
                )

                def toggle_custom_retrieval_model_path(retrieval_model_name):
                    return gr.update(visible=retrieval_model_name == "custom")

                retrieval_model_name.change(
                    toggle_custom_retrieval_model_path,
                    inputs=retrieval_model_name,
                    outputs=custom_retrieval_model_path
                )

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        inputfiles = gr.File(label=get_text("en", "upload_files"), file_count="multiple", file_types=["image"], elem_classes="image-upload")

                    with gr.Group():
                        with gr.Accordion(get_text("en", "optimization_params"), open=True) as opt_params_accordion:
                            with gr.Row():
                                lr1 = gr.Slider(label=get_text("en", "coarse_lr"), value=0.07, minimum=0.01, maximum=0.2, step=0.01, info=get_text("en", "coarse_lr_info"))
                                niter1 = gr.Slider(label=get_text("en", "coarse_iter"), value=300, minimum=0, maximum=1000, step=1, info=get_text("en", "coarse_iter_info"))
                            with gr.Row():
                                lr2 = gr.Slider(label=get_text("en", "fine_lr"), value=0.01, minimum=0.005, maximum=0.05, step=0.001, info=get_text("en", "fine_lr_info"))
                                niter2 = gr.Slider(label=get_text("en", "fine_iter"), value=300, minimum=0, maximum=1000, step=1, info=get_text("en", "fine_iter_info"))
                            optim_level = gr.Dropdown(["coarse", "refine", "refine+depth"], value='refine+depth', label=get_text("en", "optim_level"), info=get_text("en", "optim_level_info"))
                            matching_conf_thr = gr.Slider(label=get_text("en", "matching_conf_thr"), value=0., minimum=0., maximum=30., step=0.1, info=get_text("en", "matching_conf_thr_info"))
                            shared_intrinsics = gr.Checkbox(value=False, label=get_text("en", "shared_intrinsics"), info=get_text("en", "shared_intrinsics_info"))

                        with gr.Accordion(get_text("en", "scenegraph_params"), open=False) as sg_params_accordion:
                            scenegraph_type = gr.Dropdown(
                                [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")],
                                value='complete', label=get_text("en", "scenegraph_type"),
                                info=get_text("en", "scenegraph_type_info"), interactive=True)
                            with gr.Column(visible=False) as graph_opt:
                                winsize = gr.Slider(label=get_text("en", "scenegraph_window_size"), value=1, minimum=1, maximum=1, step=1)
                                win_cyclic = gr.Checkbox(value=False, label=get_text("en", "cyclic_sequence"))
                                refid = gr.Slider(label=get_text("en", "reference_id"), value=0, minimum=0, maximum=0, step=1, visible=False)

                        with gr.Accordion(get_text("en", "visualization_params"), open=False) as viz_params_accordion:
                            with gr.Row():
                                min_conf_thr = gr.Slider(label=get_text("en", "min_conf_thr"), value=1.5, minimum=0.0, maximum=10, step=0.1, info=get_text("en", "min_conf_thr_info"))
                                cam_size = gr.Slider(label=get_text("en", "cam_size"), value=0.2, minimum=0.001, maximum=1.0, step=0.001, info=get_text("en", "cam_size_info"))
                            TSDF_thresh = gr.Slider(label=get_text("en", "tsdf_threshold"), value=0., minimum=0., maximum=1., step=0.01, info=get_text("en", "tsdf_threshold_info"))
                            with gr.Row():
                                as_pointcloud = gr.Checkbox(value=True, label=get_text("en", "as_pointcloud"))
                                mask_sky = gr.Checkbox(value=False, label=get_text("en", "mask_sky"))
                                clean_depth = gr.Checkbox(value=True, label=get_text("en", "clean_depth"))
                                transparent_cams = gr.Checkbox(value=False, label=get_text("en", "transparent_cams"))
                        export_format = gr.Dropdown([".glb", ".obj", ".ply", ".stl"], value=".glb", label="Export Format")

                    run_btn = gr.Button(get_text("en", "run"), variant="primary")

                with gr.Column(scale=2):
                    with gr.Group():
                        outmodel = gr.Model3D(label="3D Model Output")

            def update_ui_text(language, sg_type, in_files, cyclic, ref_id, retrieval_model_name_val):
                graph_opt_up, winsize_up, win_cyclic_up, refid_up = set_scenegraph_options(in_files, cyclic, ref_id, sg_type)

                scenegraph_type_choices = [
                    (get_text(language, "sg_complete"), "complete"),
                    (get_text(language, "sg_swin"), "swin"),
                    (get_text(language, "sg_logwin"), "logwin"),
                    (get_text(language, "sg_oneref"), "oneref")
                ]
                if has_retrieval and retrieval_model_name_val != get_text(language, "retrieval_model_none"):
                    scenegraph_type_choices.insert(1, (get_text(language, "sg_retrieval"), "retrieval"))

                return [
                    language,
                    gr.HTML(value=f'<h2 id="title" style="text-align: left; flex-grow: 1; margin: 0;">{get_text(language, "title")}</h2>'),
                    gr.File(label=get_text(language, "upload_files")),
                    gr.Button(value=get_text(language, "run")),
                    gr.Accordion(label=get_text(language, "optimization_params")),
                    gr.Slider(label=get_text(language, "coarse_lr"), info=get_text(language, "coarse_lr_info")),
                    gr.Slider(label=get_text(language, "coarse_iter"), info=get_text(language, "coarse_iter_info")),
                    gr.Slider(label=get_text(language, "fine_lr"), info=get_text(language, "fine_lr_info")),
                    gr.Slider(label=get_text(language, "fine_iter"), info=get_text(language, "fine_iter_info")),
                    gr.Dropdown(label=get_text(language, "optim_level"), info=get_text(language, "optim_level_info")),
                    gr.Slider(label=get_text(language, "matching_conf_thr"), info=get_text(language, "matching_conf_thr_info")),
                    gr.Checkbox(label=get_text(language, "shared_intrinsics"), info=get_text(language, "shared_intrinsics_info")),
                    gr.Accordion(label=get_text(language, "scenegraph_params")),
                    gr.Dropdown(choices=scenegraph_type_choices, label=get_text(language, "scenegraph_type"), info=get_text(language, "scenegraph_type_info")),
                    winsize_up,
                    gr.Checkbox(label=get_text(language, "cyclic_sequence")),
                    refid_up,
                    graph_opt_up,
                    gr.Accordion(label=get_text(language, "visualization_params")),
                    gr.Slider(label=get_text(language, "min_conf_thr"), info=get_text(language, "min_conf_thr_info")),
                    gr.Slider(label=get_text(language, "cam_size"), info=get_text(language, "cam_size_info")),
                    gr.Slider(label=get_text(language, "tsdf_threshold"), info=get_text(language, "tsdf_threshold_info")),
                    gr.Checkbox(label=get_text(language, "as_pointcloud")),
                    gr.Checkbox(label=get_text(language, "mask_sky")),
                    gr.Checkbox(label=get_text(language, "clean_depth")),
                    gr.Checkbox(label=get_text(language, "transparent_cams")),
                    gr.Accordion(label=get_text(language, "config_title")),
                    gr.Radio(label=get_text(language, "device")),
                    gr.Dropdown(label=get_text(language, "model")),
                    gr.Textbox(label=get_text(language, "custom_model_path"), placeholder=get_text(language, "custom_model_path_placeholder")),
                    gr.Dropdown(
                        choices=[
                            get_text(language, "retrieval_model_none"),
                            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree",
                            "custom"
                        ],
                        label=get_text(language, "retrieval_model")
                    ),
                    gr.Textbox(
                        label=get_text(language, "custom_retrieval_model_path"),
                        placeholder=get_text(language, "custom_retrieval_model_path_placeholder")
                    ),
                ]

            ui_components = [
                lang_state, title_html, inputfiles, run_btn,
                opt_params_accordion, lr1, niter1, lr2, niter2, optim_level, matching_conf_thr, shared_intrinsics,
                sg_params_accordion, scenegraph_type, winsize, win_cyclic, refid, graph_opt,
                viz_params_accordion, min_conf_thr, cam_size, TSDF_thresh, as_pointcloud, mask_sky, clean_depth, transparent_cams,
                config_accordion, device, model_name, custom_model_path,
                retrieval_model_name, custom_retrieval_model_path
            ]

            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[graph_opt, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            run_btn.click(fn=run_reconstruction,
                          inputs=[scene_state, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics,
                                  model_name, custom_model_path, device, export_format,
                                  retrieval_model_name, custom_retrieval_model_path],
                          outputs=[scene_state, outmodel])

            viz_inputs = [scene_state, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, TSDF_thresh, export_format]
            min_conf_thr.release(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            cam_size.release(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            TSDF_thresh.release(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            transparent_cams.change(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)
            export_format.change(fn=model_from_scene_fun, inputs=viz_inputs, outputs=outmodel)

            def on_lang_change(language_name):
                lang_code = "zh" if language_name == "中文" else "en"
                return lang_code

            lang_radio.change(on_lang_change, inputs=lang_radio, outputs=lang_state, queue=False).then(
                update_ui_text,
                inputs=[lang_state, scenegraph_type, inputfiles, win_cyclic, refid, retrieval_model_state],
                outputs=ui_components
            )

            retrieval_model_name.change(lambda x: x, inputs=retrieval_model_name, outputs=retrieval_model_state, queue=False).then(
                update_ui_text,
                inputs=[lang_state, scenegraph_type, inputfiles, win_cyclic, refid, retrieval_model_state],
                outputs=ui_components
            )

        demo.launch(share=args.share, server_name=server_name, server_port=args.server_port)


if __name__ == '__main__':
    parser = get_args_parser()

    for group in parser._mutually_exclusive_groups:
        if any(action.dest in ['model_name', 'weights'] for action in group._group_actions):
            group.required = False
            break

    args = parser.parse_args()
    set_print_with_timestamp()
    main(args)
