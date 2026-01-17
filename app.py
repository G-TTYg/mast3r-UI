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

from mast3r.demo import get_args_parser, get_reconstructed_scene, get_3D_model_from_scene, set_scenegraph_options, SparseGAState
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True

# --- Localization Dictionary ---
i18n = {
    "en": {
        "title": "MASt3R Enhanced UI",
        "upload_gallery": "Image Gallery",
        "run": "Run Reconstruction",
        # New Configuration section
        "config_title": "Configuration",
        "device": "Device",
        "model": "Model",
        "retrieval_model": "Retrieval Model",
        "retrieval_model_info": "Optional model to accelerate pair selection for large scenes.",
        "custom_model_path": "Custom Model Path",
        "custom_model_path_placeholder": "Enter path to your custom .pth file",
        "custom_retrieval_model_path": "Custom Retrieval Model Path",
        "custom_retrieval_model_path_placeholder": "Enter path to your custom retrieval .pth file",
        "none": "None",

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
        "output_format": "Output Format",
        "output_format_info": "Choose the format of the 3D model output.",
        "point_cloud": "Point Cloud",
        "mesh": "Mesh",
        "mask_sky": "Mask Sky",
        "clean_depth": "Clean-up Depth Maps",
        "transparent_cams": "Transparent Cameras",
    },
    "zh": {
        "title": "MASt3R 增强版UI",
        "upload_gallery": "图像画廊",
        "run": "开始重建",
        # New Configuration section
        "config_title": "配置",
        "device": "设备",
        "model": "模型",
        "retrieval_model": "检索模型",
        "retrieval_model_info": "可选的模型，用于加速大型场景的图像对选择。",
        "custom_model_path": "自定义模型路径",
        "custom_model_path_placeholder": "请输入您的自定义 .pth 文件路径",
        "custom_retrieval_model_path": "自定义检索模型路径",
        "custom_retrieval_model_path_placeholder": "请输入您的自定义检索 .pth 文件路径",
        "none": "无",

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
        "output_format": "输出格式",
        "output_format_info": "选择三维模型的输出格式。",
        "point_cloud": "点云",
        "mesh": "网格",
        "mask_sky": "遮蔽天空",
        "clean_depth": "清理深度图",
        "transparent_cams": "透明相机",
    }
}

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
        "device": None
    }

    def get_model(model_name, custom_model_path, device):
        if model_name == "custom":
            weights_path = custom_model_path
        else:
            weights_path = "naver/" + model_name

        # Check if the model is already loaded with the correct weights and device
        if (model_cache["model"] is not None and
            model_cache["weights_path"] == weights_path and
            model_cache["device"] == device):
            return model_cache["model"]

        # Load the model
        model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)

        # Update the cache
        model_cache["model"] = model
        model_cache["weights_path"] = weights_path
        model_cache["device"] = device

        return model

    def run_reconstruction(scene_state, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                           output_format_r, mask_sky, clean_depth, transparent_cams, cam_size,
                           scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics,
                           model_name_dd, custom_model_path_tb, retrieval_model_name_dd, custom_retrieval_model_path_tb,
                           device_r, lang, progress=gr.Progress()):

        progress(0, desc="Loading model...")
        model = get_model(model_name_dd, custom_model_path_tb, device_r)

        # Extract file paths from the Gallery's output
        if inputfiles:
            inputfiles = [f['name'] for f in inputfiles]

        if retrieval_model_name_dd == "custom":
            retrieval_model_path = custom_retrieval_model_path_tb
        elif retrieval_model_name_dd == "none":
            retrieval_model_path = None
        else:
            retrieval_model_path = "naver/" + retrieval_model_name_dd

        chkpt_tag = hash_md5(model_cache["weights_path"])

        # Use the persistent temporary directory created at the start of main()
        cache_path = os.path.join(args.tmp_dir, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)

        recon_fun = functools.partial(get_reconstructed_scene, cache_path, args.gradio_delete_cache, model,
                                        retrieval_model_path, device_r, args.silent, args.image_size)

        as_pointcloud = (output_format_r == get_text(lang, "point_cloud"))

        progress(0.1, desc="Running reconstruction...")
        scene_state, outmodel = recon_fun(scene_state, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                          as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                          scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics)
        progress(1.0, desc="Done!")
        return scene_state, outmodel

    # Create a persistent temporary directory for the duration of the app
    with tempfile.TemporaryDirectory(suffix='_mast3r_gradio_persistent_cache') as tmpdirname:
        # Pass this persistent directory to the reconstruction function via args
        args.tmp_dir = tmpdirname

        model_from_scene_fun = functools.partial(get_3D_model_from_scene, args.silent)

        # Build Gradio UI
        with gr.Blocks(css=".gradio-container {margin: 0 !important; min-width: 100%}", title="MASt3R Enhanced UI") as demo:
            scene_state = gr.State(None)
            lang_state = gr.State("en") # Default language

            with gr.Row():
                title_html = gr.HTML('<h2 id="title" style="text-align: left; flex-grow: 1; margin: 0;">MASt3R Enhanced UI</h2>')
                lang_radio = gr.Radio(["English", "中文"], value="English", label="Language", show_label=False, container=False, scale=0)

            with gr.Accordion("Configuration", open=True) as config_accordion:
                device = gr.Radio(["cpu", "cuda"], value="cpu" if not torch.cuda.is_available() else "cuda", label="Device")
                with gr.Row():
                    model_name = gr.Dropdown(["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", "custom"], label="Model", scale=1)
                    retrieval_model_name = gr.Dropdown(["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree", "custom", "none"], label="Retrieval Model", info="Optional model to accelerate pair selection for large scenes.", scale=1)
                with gr.Row():
                    custom_model_path = gr.Textbox(label="Custom Model Path", placeholder="Enter path to your custom .pth file", visible=False, scale=1)
                    custom_retrieval_model_path = gr.Textbox(label="Custom Retrieval Model Path", placeholder="Enter path to your custom retrieval .pth file", visible=False, scale=1)

                def toggle_custom_paths(model_name, retrieval_model_name):
                    return gr.update(visible=model_name == "custom"), gr.update(visible=retrieval_model_name == "custom")

                model_name.change(toggle_custom_paths, inputs=[model_name, retrieval_model_name], outputs=[custom_model_path, custom_retrieval_model_path])
                retrieval_model_name.change(toggle_custom_paths, inputs=[model_name, retrieval_model_name], outputs=[custom_model_path, custom_retrieval_model_path])

                def update_scenegraph_options(retrieval_model_name, lang):
                    choices = [
                        (get_text(lang, "sg_complete"), "complete"),
                        (get_text(lang, "sg_swin"), "swin"),
                        (get_text(lang, "sg_logwin"), "logwin"),
                        (get_text(lang, "sg_oneref"), "oneref")
                    ]
                    if retrieval_model_name != "none":
                        choices.insert(1, (get_text(lang, "sg_retrieval"), "retrieval"))
                    return gr.update(choices=choices)

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        inputfiles = gr.Gallery(label=get_text("en", "upload_gallery"))

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

                    with gr.Group():
                        with gr.Accordion(get_text("en", "scenegraph_params"), open=False) as sg_params_accordion:
                            scenegraph_type = gr.Dropdown(
                                [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")],
                                value='complete', label=get_text("en", "scenegraph_type"),
                                info=get_text("en", "scenegraph_type_info"), interactive=True)
                            retrieval_model_name.change(update_scenegraph_options, inputs=[retrieval_model_name, lang_state], outputs=scenegraph_type)
                            with gr.Column(visible=False) as graph_opt:
                                winsize = gr.Slider(label=get_text("en", "scenegraph_window_size"), value=1, minimum=1, maximum=1, step=1)
                                win_cyclic = gr.Checkbox(value=False, label=get_text("en", "cyclic_sequence"))
                                refid = gr.Slider(label=get_text("en", "reference_id"), value=0, minimum=0, maximum=0, step=1, visible=False)

                    with gr.Group():
                        with gr.Accordion(get_text("en", "visualization_params"), open=False) as viz_params_accordion:
                            with gr.Row():
                                min_conf_thr = gr.Slider(label=get_text("en", "min_conf_thr"), value=1.5, minimum=0.0, maximum=10, step=0.1, info=get_text("en", "min_conf_thr_info"))
                                cam_size = gr.Slider(label=get_text("en", "cam_size"), value=0.2, minimum=0.001, maximum=1.0, step=0.001, info=get_text("en", "cam_size_info"))
                            TSDF_thresh = gr.Slider(label=get_text("en", "tsdf_threshold"), value=0., minimum=0., maximum=1., step=0.01, info=get_text("en", "tsdf_threshold_info"))
                            output_format = gr.Radio([get_text("en", "point_cloud"), get_text("en", "mesh")], value=get_text("en", "point_cloud"), label=get_text("en", "output_format"), info=get_text("en", "output_format_info"))
                            with gr.Row():
                                mask_sky = gr.Checkbox(value=False, label=get_text("en", "mask_sky"))
                                clean_depth = gr.Checkbox(value=True, label=get_text("en", "clean_depth"))
                                transparent_cams = gr.Checkbox(value=False, label=get_text("en", "transparent_cams"))

                    run_btn = gr.Button(get_text("en", "run"), variant="primary")

                with gr.Column(scale=2):
                    with gr.Group():
                        outmodel = gr.Model3D(label="3D Model Output", height=600)

            # Language switching logic
            def update_ui_text(language, sg_type, in_files, cyclic, ref_id, retrieval_model_name):
                # Update scene graph options based on language as well
                graph_opt_up, winsize_up, win_cyclic_up, refid_up = set_scenegraph_options(in_files, cyclic, ref_id, sg_type)

                # Dynamically get text for dropdown
                scenegraph_type_choices = [
                    (get_text(language, "sg_complete"), "complete"),
                    (get_text(language, "sg_swin"), "swin"),
                    (get_text(language, "sg_logwin"), "logwin"),
                    (get_text(language, "sg_oneref"), "oneref")
                ]
                if retrieval_model_name != "none":
                    scenegraph_type_choices.insert(1, (get_text(language, "sg_retrieval"), "retrieval"))

                return [
                    language,
                    gr.HTML(value=f'<h2 id="title" style="text-align: left; flex-grow: 1; margin: 0;">{get_text(language, "title")}</h2>'),
                    gr.Gallery(label=get_text(language, "upload_gallery")),
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
                    gr.Radio(choices=[get_text(language, "point_cloud"), get_text(language, "mesh")], label=get_text(language, "output_format"), info=get_text(language, "output_format_info")),
                    gr.Checkbox(label=get_text(language, "mask_sky")),
                    gr.Checkbox(label=get_text(language, "clean_depth")),
                    gr.Checkbox(label=get_text(language, "transparent_cams")),
                    gr.Accordion(label=get_text(language, "config_title")),
                    device.update(label=get_text(language, "device")),
                    gr.Dropdown(label=get_text(language, "model")),
                    gr.Dropdown(label=get_text(language, "retrieval_model"), info=get_text(language, "retrieval_model_info")),
                    gr.Textbox(label=get_text(language, "custom_model_path"), placeholder=get_text(language, "custom_model_path_placeholder")),
                    gr.Textbox(label=get_text(language, "custom_retrieval_model_path"), placeholder=get_text(language, "custom_retrieval_model_path_placeholder")),
                ]

            # Collect all components that need updating
            ui_components = [
                lang_state, title_html, inputfiles, run_btn,
                opt_params_accordion, lr1, niter1, lr2, niter2, optim_level, matching_conf_thr, shared_intrinsics,
                sg_params_accordion, scenegraph_type, winsize, win_cyclic, refid, graph_opt,
                viz_params_accordion, min_conf_thr, cam_size, TSDF_thresh, output_format, mask_sky, clean_depth, transparent_cams,
                config_accordion, device, model_name, retrieval_model_name, custom_model_path, custom_retrieval_model_path
            ]

            def model_from_scene_wrapper(scene, min_conf_thr, output_format_r, mask_sky,
                                     clean_depth, transparent_cams, cam_size, TSDF_thresh, lang):
                as_pointcloud = (output_format_r == get_text(lang, "point_cloud"))
                return model_from_scene_fun(scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh)

            # Event listeners
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
                                  output_format, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics,
                                  model_name, custom_model_path, retrieval_model_name, custom_retrieval_model_path,
                                  device, lang_state],
                          outputs=[scene_state, outmodel])

            # Listen to changes in visualization parameters
            viz_inputs = [scene_state, min_conf_thr, output_format, mask_sky, clean_depth, transparent_cams, cam_size, TSDF_thresh, lang_state]
            min_conf_thr.release(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)
            cam_size.release(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)
            TSDF_thresh.release(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)
            output_format.change(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)
            mask_sky.change(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)
            clean_depth.change(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)
            transparent_cams.change(fn=model_from_scene_wrapper, inputs=viz_inputs, outputs=outmodel)

            # Bind the radio button to the update function
            def on_lang_change(language_name):
                lang_code = "zh" if language_name == "中文" else "en"
                return lang_code

            lang_radio.change(on_lang_change, inputs=lang_radio, outputs=lang_state, queue=False).then(
                update_ui_text,
                inputs=[lang_state, scenegraph_type, inputfiles, win_cyclic, refid, retrieval_model_name],
                outputs=ui_components
            )

        demo.launch(share=args.share, server_name=server_name, server_port=args.server_port)


if __name__ == '__main__':
    parser = get_args_parser()

    # Find the mutually exclusive group containing --model_name and --weights and make it not required.
    for group in parser._mutually_exclusive_groups:
        if any(action.dest in ['model_name', 'weights'] for action in group._group_actions):
            group.required = False
            break

    args = parser.parse_args()
    set_print_with_timestamp()
    main(args)
