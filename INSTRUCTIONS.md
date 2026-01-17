# MASt3R Enhanced UI - English User Guide

Welcome to the MASt3R Enhanced UI! This is a Gradio-based graphical interface designed to make using the powerful features of MASt3R easier and more intuitive.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Model Downloads](#2-model-downloads)
3. [Launching the UI](#3-launching-the-ui)
4. [Interface Guide](#4-interface-guide)
   - [File Upload](#file-upload)
   - [Parameter Adjustment](#parameter-adjustment)
   - [Running & Viewing Results](#running--viewing-results)
5. [Special Instructions for Windows Users](#5-special-instructions-for-windows-users)

---

### 1. Environment Setup

We highly recommend using Conda to create an isolated Python environment to avoid conflicts with other libraries on your system.

If you haven't installed Conda, please download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

**Steps:**

1.  **Create a Conda Environment**:
    Open your terminal (or `Anaconda Prompt` on Windows) and run the following command to create a new environment named `mast3r`. We recommend using Python 3.11.

    ```bash
    conda create -n mast3r python=3.11
    ```

2.  **Activate the Conda Environment**:
    Once created, activate the new environment using the command below. Ensure this environment is activated before all subsequent operations.

    ```bash
    conda activate mast3r
    ```

3.  **Install Dependencies**:
    In the activated environment, execute the following two commands to install all required dependencies.

    ```bash
    # Install main dependencies
    pip install -r requirements.txt

    # Install dependencies for the dust3r module
    pip install -r dust3r/requirements.txt
    ```

---

### 2. Model Downloads

The application requires pre-trained model files to function.

#### a) Main Reconstruction Model

- **Automatic Download**: By default, the program will automatically download the `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` model from Hugging Face Hub. On the first run, please ensure you have a stable internet connection and wait patiently for the download to complete.

- **Manual Download**: If the automatic download is slow or fails, you can find the manual download link in the main `README.md` file. After downloading, we recommend placing the model file (e.g., `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`) in a `checkpoints/` folder in the project's root directory (create the folder if it doesn't exist). You can then specify the path to this file in the UI.

#### b) Retrieval Model (Optional, but Recommended for >50 images)

The retrieval model is an optional component that significantly speeds up the reconstruction process for large scenes by intelligently pre-selecting which image pairs to match.

- **Download Links**:
  You must download **both** of the following files:
  - [Model Weights (.pth)](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth)
  - [Codebook (.pkl)](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl)

- **Setup**:
  1. Create a `checkpoints` folder in the root of the project if it doesn't already exist.
  2. Place **both** the downloaded `.pth` and `.pkl` files into the `checkpoints` folder.
  3. In the UI's "Configuration" section, select the "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree" option from the "Retrieval Model" dropdown. If you place it elsewhere, you may select "custom" and specify the full path to the `.pth` file in the "Custom Retrieval Model Path" textbox. The application will automatically look for the `.pkl` file in the same directory.

---

### 3. Launching the UI

After setting up the environment, you can launch the UI.

In your terminal, execute the following command:

```bash
python app.py
```

Upon successful launch, you will see output similar to this in your terminal:
`Running on local URL:  http://127.0.0.1:7860`

Copy this URL and open it in your web browser to access the application interface.

---

### 4. Interface Guide

#### File Upload
In the "Upload Image Files" area on the left, you can click to browse or drag and drop multiple image files. These images are the basis for the 3D reconstruction.

#### Parameter Adjustment
Below the file upload area, there are three collapsible sections for parameter adjustments:

- **Optimization Parameters**:
  These are core parameters that control the model's optimization process, such as learning rates (LR) and iteration counts for coarse and fine alignment. Adjusting these can affect the speed and accuracy of the reconstruction.

- **Scene Graph Parameters**:
  Here you can choose different strategies for matching image pairs. The `complete` mode attempts to match all possible pairs, which is thorough but time-consuming. Modes like `swin` (sliding window) are better suited for ordered image sequences (like video frames). Using a `retrieval` model is best for large, unordered collections.

- **Visualization Parameters**:
  These parameters control the visual output of the 3D model. You can adjust `min_conf_thr` to filter out low-confidence points or change `cam_size` to alter the size of the camera frustums in the preview. Check `As pointcloud` to display the result as a point cloud instead of a mesh.

#### Running & Viewing Results
Once all parameters are set, click the blue **"Run"** button to start the reconstruction. The process may take some time, depending on the number of images, their resolution, and your hardware.

When complete, the 3D model viewer on the right will display the result. You can use your mouse to rotate, zoom, and pan the model.

---

### 5. Special Instructions for Windows Users

For convenience, a one-click startup script, `start_ui.bat`, is provided for Windows users.

Simply double-click this script to automatically execute the startup command using the CPU. A command prompt window will remain open, where you can view the program's logs and find the access URL.
