
# MASt3R Enhanced UI - User Guide

Welcome to the MASt3R Enhanced UI! This is a Gradio-based graphical interface designed to make the powerful features of MASt3R more accessible and intuitive.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Model Download](#2-model-download)
3. [Launching the UI](#3-launching-the-ui)
4. [How to Use the Interface](#4-how-to-use-the-interface)
   - [File Upload](#file-upload)
   - [Parameter Adjustment](#parameter-adjustment)
   - [Running and Viewing Results](#running-and-viewing-results)
5. [Special Note for Windows Users](#5-special-note-for-windows-users)

---

### 1. Environment Setup

We strongly recommend using Conda to create an isolated Python environment. This will prevent conflicts with other libraries on your system.

If you don't have Conda installed, please download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

**Follow these steps:**

1.  **Create a Conda Environment**:
    Open your terminal (or `Anaconda Prompt` on Windows) and run the following command to create a new environment named `mast3r`. We recommend using Python 3.11.

    ```bash
    conda create -n mast3r python=3.11
    ```

2.  **Activate the Conda Environment**:
    Once created, activate the new environment using the command below. Ensure this environment is activated before performing any subsequent steps.

    ```bash
    conda activate mast3r
    ```

3.  **Install Dependencies**:
    With the environment activated, run the following two commands to install all required dependencies.

    ```bash
    # Install main dependencies
    pip install -r requirements.txt

    # Install dependencies for the dust3r module
    pip install -r dust3r/requirements.txt
    ```

---

### 2. Model Download

This application requires a pre-trained model file to run.

- **Automatic Download**:
  By default, the program will automatically download the `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` model from the Hugging Face Hub. On your first run, please ensure you have a stable internet connection and allow some time for the download to complete.

- **Manual Download**:
  If the automatic download is slow or fails, you can find manual download links for the model in the `README.md` file. After downloading, place the model file (e.g., `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`) into a `checkpoints/` folder in the project's root directory (create the folder if it doesn't exist).

  If the automatic download is slow or fails, you can find manual download links for the model in the `README.md` file. After downloading, we recommend placing the model file into a `checkpoints/` folder in the project's root directory (create the folder if it doesn't exist). You will be able to specify the exact path to this file within the UI later.

---

### 3. Launching the UI

Once the environment setup is complete, you can launch the UI.

In your terminal, execute the following command:

```bash
python app.py
```

Upon successful launch, you will see output in your terminal similar to this:
`Running on local URL:  http://127.0.0.1:7860`

Copy this URL and open it in your web browser to access the application's UI.

---

### 4. How to Use the Interface

#### File Upload
On the left-hand side, in the "Upload Image Files" area, you can click to browse or drag and drop multiple image files. These images are the basis for the 3D reconstruction.

#### Parameter Adjustment
Below the file upload area, there are three collapsible sections for parameter adjustments:

- **Optimization Parameters**:
  This section contains the core parameters that control the model's optimization process, such as the learning rates (LR) and number of iterations for both coarse and fine alignment. Adjusting these can impact the speed and accuracy of the reconstruction.

- **Scene Graph Parameters**:
  Here you can choose different strategies for how image pairs are matched. The `complete` mode attempts to match all possible pairs, which is thorough but time-consuming. Other modes like `swin` (sliding window) are better suited for ordered sequences of images (e.g., video frames).

- **Visualization Parameters**:
  These parameters control the visual appearance of the final 3D model. You can adjust `min_conf_thr` to filter out low-confidence points or change `cam_size` to alter the size of the camera frustums in the preview. Check `As pointcloud` to render the output as a point cloud instead of a mesh.

#### Running and Viewing Results
After setting all parameters, click the blue **"Run"** button to start the reconstruction. The process may take some time, depending on the number of images, their resolution, and your machine's performance.

Once complete, the 3D model will be displayed in the preview panel on the right. You can interact with the model by clicking and dragging to rotate, zoom, and pan.

---

### 5. Special Note for Windows Users

To simplify the process for Windows users, a one-click startup script named `start_ui.bat` is provided.

Simply double-click this script, and it will automatically execute the command to launch the UI in CPU mode. A command prompt window will remain open, where you can view the application's logs and find the access URL.
