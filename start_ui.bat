
@echo off
REM MASt3R 增强版UI Windows 启动脚本
REM ==================================================
REM 使用说明:
REM 1. 确保您已经按照 INSTRUCTIONS.zh.md 文档中的说明安装了所有依赖。
REM 2. 将此脚本放置在项目的根目录下。
REM 3. 双击运行此脚本即可启动UI界面。
REM ==================================================

echo 正在启动 MASt3R 增强版UI...
echo.
echo 默认使用 CPU 模式运行。如果您拥有并希望使用 NVIDIA GPU，
echo 请手动编辑此文件，删除 "--device cpu" 参数。
echo.

REM 默认模型名称
set MODEL_NAME=MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

REM 启动 app.py。默认使用CPU以保证最大兼容性。
python app.py --model_name %MODEL_NAME% --device cpu

echo.
echo 程序已退出。
echo 按任意键关闭此窗口...
pause > nul
