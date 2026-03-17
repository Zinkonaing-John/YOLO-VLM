"""Generate PDF training guides for YOLO+CLIP and CNN+ResNet pipelines."""

from fpdf import FPDF

# ─── Shared styles ────────────────────────────────────────────────────


class GuidePDF(FPDF):
    DARK = (24, 24, 27)
    WHITE = (244, 244, 245)
    ACCENT = (139, 92, 246)   # purple
    ACCENT2 = (249, 115, 22)  # orange
    GREEN = (16, 185, 129)
    RED = (239, 68, 68)
    GRAY = (161, 161, 170)
    CODE_BG = (39, 39, 42)

    def __init__(self, accent=None):
        super().__init__()
        self.accent = accent or self.ACCENT
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.GRAY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def cover_page(self, title, subtitle, pipeline_color):
        self.add_page()
        self.set_fill_color(*self.DARK)
        self.rect(0, 0, 210, 297, "F")

        self.ln(60)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(*pipeline_color)
        self.cell(0, 15, title, align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(5)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*self.GRAY)
        self.cell(0, 10, subtitle, align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(15)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*self.WHITE)
        self.cell(0, 8, "Industrial AI Vision Inspector", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Training & Fine-Tuning Guide", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(60)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*self.GRAY)
        self.cell(0, 8, "Project: vlm-yolo", align="C", new_x="LMARGIN", new_y="NEXT")

    def section_title(self, num, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.accent)
        self.cell(0, 10, f"  {num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.accent)
        self.set_line_width(0.5)
        x = self.get_x() + 10
        y = self.get_y()
        self.line(x, y, x + 180, y)
        self.ln(4)

    def sub_title(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.WHITE)
        self.cell(0, 8, f"    {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.WHITE)
        self.multi_cell(0, 5.5, f"    {text}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet(self, text, indent=12):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.WHITE)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, f" {text}", new_x="LMARGIN", new_y="NEXT")

    def code_block(self, code):
        self.ln(2)
        self.set_font("Courier", "", 9)
        self.set_fill_color(*self.CODE_BG)
        self.set_text_color(200, 200, 200)
        x = self.get_x() + 12
        w = 180
        lines = code.strip().split("\n")
        h = len(lines) * 5 + 6

        if self.get_y() + h > 270:
            self.add_page()

        y = self.get_y()
        self.rect(x, y, w, h, "F")
        self.set_xy(x + 4, y + 3)
        for line in lines:
            self.set_x(x + 4)
            self.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def step(self, num, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.GREEN)
        self.cell(0, 7, f"      Step {num}: {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def warning(self, text):
        self.ln(1)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.RED)
        self.multi_cell(0, 5.5, f"    WARNING: {text}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def tip(self, text):
        self.ln(1)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*self.GREEN)
        self.multi_cell(0, 5.5, f"    TIP: {text}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def new_page(self):
        self.add_page()
        self.set_fill_color(*self.DARK)
        self.rect(0, 0, 210, 297, "F")


# ═════════════════════════════════════════════════════════════════════
#  GUIDE 1: YOLO + CLIP
# ═════════════════════════════════════════════════════════════════════

def generate_yolo_clip_guide():
    pdf = GuidePDF(accent=GuidePDF.ACCENT)
    pdf.alias_nb_pages()
    pdf.cover_page(
        "YOLO + CLIP Pipeline",
        "Training & Fine-Tuning Guide",
        GuidePDF.ACCENT,
    )

    # ── Section 1: Overview ──
    pdf.new_page()
    pdf.section_title(1, "Pipeline Overview")
    pdf.body_text(
        "The YOLO+CLIP pipeline combines two AI models for industrial defect detection. "
        "YOLOv8 handles object/defect localization with bounding boxes, while CLIP provides "
        "zero-shot semantic classification of each detected region."
    )
    pdf.sub_title("Pipeline Flow")
    pdf.body_text("Camera -> Image -> Contour Detection -> CLIP Object Labeling -> YOLO Defect Detection -> Per-ROI CLIP Classification -> OK/NG Verdict -> Save to DB -> Dashboard")

    pdf.sub_title("Models Used")
    pdf.bullet("YOLOv8n (fine-tuned) - Defect detection with bounding boxes")
    pdf.bullet("YOLOv8m (pretrained COCO) - General object detection (optional)")
    pdf.bullet("OpenAI CLIP ViT-B/32 - Zero-shot defect classification")

    pdf.sub_title("What You Will Learn")
    pdf.bullet("How to prepare a defect detection dataset in YOLO format")
    pdf.bullet("How to fine-tune YOLOv8 on your custom defect data")
    pdf.bullet("How to configure CLIP labels for your specific defect types")
    pdf.bullet("How to evaluate and deploy the trained models")

    # ── Section 2: Prerequisites ──
    pdf.new_page()
    pdf.section_title(2, "Prerequisites")

    pdf.sub_title("Hardware Requirements")
    pdf.bullet("GPU recommended: NVIDIA (CUDA) or Apple Silicon (MPS)")
    pdf.bullet("Minimum 8GB RAM, 16GB recommended")
    pdf.bullet("At least 5GB free disk space for datasets and weights")

    pdf.sub_title("Software Requirements")
    pdf.code_block(
        "# Python 3.10+\n"
        "pip install ultralytics==8.3.57\n"
        "pip install torch>=2.1.0 torchvision>=0.16.0\n"
        "pip install git+https://github.com/openai/CLIP.git\n"
        "pip install opencv-python-headless numpy<2.0.0\n"
        "pip install pyyaml kagglehub  # for dataset download"
    )

    pdf.sub_title("Project Structure")
    pdf.code_block(
        "vlm-yolo/\n"
        "  backend/\n"
        "    train.py                    # YOLO training script\n"
        "    datasets/NEU-DET/           # Dataset directory\n"
        "    weights/best.pt             # Trained defect model\n"
        "    app/core/config.py          # Model configuration\n"
        "    app/models/ai_models.py     # YOLO & CLIP wrappers\n"
        "  weights/                      # Shared weights directory"
    )

    # ── Section 3: Dataset Preparation ──
    pdf.new_page()
    pdf.section_title(3, "Dataset Preparation (YOLO)")

    pdf.sub_title("3.1 Supported Dataset: NEU Steel Surface Defects")
    pdf.body_text(
        "The included training script uses the NEU-DET dataset from Kaggle, which contains "
        "1,800 training images and 360 validation images across 6 steel surface defect classes."
    )
    pdf.bullet("crazing - Fine network of surface cracks")
    pdf.bullet("inclusion - Foreign material embedded in surface")
    pdf.bullet("patches - Irregular surface patches")
    pdf.bullet("pitted_surface - Small pits/holes in surface")
    pdf.bullet("rolled-in_scale - Scale pressed into surface during rolling")
    pdf.bullet("scratches - Linear surface scratches")

    pdf.step(1, "Download the NEU-DET dataset")
    pdf.code_block(
        "# Option A: Automatic download via training script\n"
        "cd backend\n"
        "python train.py\n"
        "# This auto-downloads from Kaggle on first run\n"
        "\n"
        "# Option B: Manual download\n"
        "pip install kagglehub\n"
        "python -c \"import kagglehub; kagglehub.dataset_download(\n"
        "    'kaustubhdikshit/neu-surface-defect-database')\""
    )

    pdf.step(2, "Understand YOLO annotation format")
    pdf.body_text(
        "Each image needs a corresponding .txt label file with one line per object:"
    )
    pdf.code_block(
        "# Format: class_id  x_center  y_center  width  height\n"
        "# All values normalized to 0-1 range\n"
        "#\n"
        "# Example (scratches class = 5):\n"
        "5 0.523000 0.491500 0.354000 0.287000\n"
        "0 0.150000 0.320000 0.120000 0.090000"
    )

    pdf.step(3, "Dataset directory structure")
    pdf.code_block(
        "datasets/NEU-DET/\n"
        "  data.yaml              # Dataset config\n"
        "  train/\n"
        "    images/              # 1,800 training images (.jpg)\n"
        "    labels/              # 1,800 label files (.txt)\n"
        "  val/\n"
        "    images/              # 360 validation images\n"
        "    labels/              # 360 label files"
    )

    pdf.step(4, "Create data.yaml configuration")
    pdf.code_block(
        "# datasets/NEU-DET/data.yaml\n"
        "path: /full/path/to/datasets/NEU-DET\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 6\n"
        "names:\n"
        "  - crazing\n"
        "  - inclusion\n"
        "  - patches\n"
        "  - pitted_surface\n"
        "  - rolled-in_scale\n"
        "  - scratches"
    )

    pdf.new_page()
    pdf.sub_title("3.2 Using Your Own Custom Dataset")
    pdf.body_text(
        "To train on your own defect images, prepare the same directory structure. "
        "You can use annotation tools like:"
    )
    pdf.bullet("CVAT (cvat.ai) - Free, web-based, exports YOLO format")
    pdf.bullet("Roboflow - Cloud-based with augmentation, exports YOLO format")
    pdf.bullet("LabelImg - Desktop tool, exports VOC XML (convert with train.py)")

    pdf.step(1, "Collect images of OK and NG parts (minimum 100+ per class)")
    pdf.step(2, "Annotate defects with bounding boxes using any tool above")
    pdf.step(3, "Export in YOLO format (class_id x_center y_center w h)")
    pdf.step(4, "Split 80% train / 20% val")
    pdf.step(5, "Create data.yaml with your class names")

    pdf.tip(
        "For best results, collect at least 300 images per defect class. "
        "Include varied lighting, angles, and background conditions."
    )

    # ── Section 4: Training YOLOv8 ──
    pdf.new_page()
    pdf.section_title(4, "Training YOLOv8 Defect Detector")

    pdf.sub_title("4.1 Quick Start (One Command)")
    pdf.code_block(
        "cd backend\n"
        "python train.py\n"
        "\n"
        "# This will:\n"
        "#   1. Download NEU-DET dataset (if not cached)\n"
        "#   2. Convert VOC annotations to YOLO format\n"
        "#   3. Fine-tune yolov8n.pt for 50 epochs\n"
        "#   4. Save best weights to weights/best.pt"
    )

    pdf.sub_title("4.2 Training Configuration")
    pdf.body_text("Key parameters in train.py that you can customize:")
    pdf.code_block(
        "BASE_WEIGHTS = \"yolov8n.pt\"    # Base model (n/s/m/l/x)\n"
        "EPOCHS = 50                     # Training epochs\n"
        "IMGSZ = 640                     # Input image size\n"
        "BATCH = 16                      # Batch size\n"
        "DEVICE = \"mps\"                  # mps / cuda / cpu"
    )

    pdf.sub_title("4.3 Understanding Base Model Sizes")
    pdf.bullet("yolov8n.pt (6MB) - Nano: fastest, good for edge devices")
    pdf.bullet("yolov8s.pt (22MB) - Small: balanced speed/accuracy")
    pdf.bullet("yolov8m.pt (50MB) - Medium: higher accuracy, slower")
    pdf.bullet("yolov8l.pt (84MB) - Large: high accuracy, needs GPU")
    pdf.bullet("yolov8x.pt (131MB) - Extra-large: best accuracy, slowest")
    pdf.tip("Start with yolov8n.pt for fast iteration, then try larger models if accuracy is insufficient.")

    pdf.new_page()
    pdf.sub_title("4.4 Step-by-Step Training Process")

    pdf.step(1, "Initialize the base model")
    pdf.code_block(
        "from ultralytics import YOLO\n"
        "\n"
        "# Load pretrained YOLOv8 nano\n"
        "model = YOLO('yolov8n.pt')"
    )

    pdf.step(2, "Start training")
    pdf.code_block(
        "model.train(\n"
        "    data='datasets/NEU-DET/data.yaml',\n"
        "    epochs=50,           # Number of training passes\n"
        "    imgsz=640,           # Image size (square)\n"
        "    batch=16,            # Images per batch\n"
        "    device='mps',        # GPU device\n"
        "    patience=20,         # Early stopping patience\n"
        "    pretrained=True,     # Use pretrained backbone\n"
        "    project='runs/detect',\n"
        "    name='train',\n"
        "    exist_ok=True,\n"
        "    save=True,\n"
        "    plots=True,          # Generate training plots\n"
        ")"
    )

    pdf.step(3, "Monitor training output")
    pdf.body_text("During training, watch these key metrics:")
    pdf.bullet("mAP50 - Mean Average Precision at IoU=0.5 (target: > 0.70)")
    pdf.bullet("mAP50-95 - mAP averaged over IoU 0.5-0.95 (target: > 0.40)")
    pdf.bullet("box_loss - Bounding box regression loss (should decrease)")
    pdf.bullet("cls_loss - Classification loss (should decrease)")

    pdf.step(4, "Copy best weights to deployment location")
    pdf.code_block(
        "import shutil\n"
        "shutil.copy2(\n"
        "    'runs/detect/train/weights/best.pt',\n"
        "    'weights/best.pt'\n"
        ")"
    )

    pdf.step(5, "Verify the trained model")
    pdf.code_block(
        "model = YOLO('weights/best.pt')\n"
        "print(f'Classes: {model.names}')\n"
        "\n"
        "# Run inference on a test image\n"
        "results = model.predict(\n"
        "    source='path/to/test_image.jpg',\n"
        "    conf=0.15,\n"
        "    save=True\n"
        ")"
    )

    # ── Section 5: CLIP Configuration ──
    pdf.new_page()
    pdf.section_title(5, "Configuring CLIP Zero-Shot Classification")

    pdf.body_text(
        "CLIP does NOT require training. Instead, you configure semantic text labels "
        "that describe OK and NG conditions. CLIP compares image features against these "
        "text descriptions to classify each region."
    )

    pdf.sub_title("5.1 How CLIP Classification Works")
    pdf.bullet("1. Crop each detected ROI from the image")
    pdf.bullet("2. Encode the ROI image with CLIP vision encoder")
    pdf.bullet("3. Encode all OK + NG text labels with CLIP text encoder")
    pdf.bullet("4. Compute cosine similarity between image and text embeddings")
    pdf.bullet("5. If sum of NG similarities >= threshold (0.5), classify as defect")

    pdf.sub_title("5.2 Default Label Configuration")
    pdf.code_block(
        "# In backend/app/core/config.py\n"
        "\n"
        "CLIP_LABELS_OK = [\n"
        "    'a photo of a smooth clean metal surface',\n"
        "    'a photo of a flawless steel product',\n"
        "    'a photo of a normal metal surface without defects',\n"
        "]\n"
        "\n"
        "CLIP_LABELS_NG = [\n"
        "    'a photo of a scratched metal surface',\n"
        "    'a photo of a cracked metal surface',\n"
        "    'a photo of a metal surface with crazing defects',\n"
        "    'a photo of a metal surface with inclusion defects',\n"
        "    'a photo of a pitted metal surface',\n"
        "    'a photo of a metal surface with rolled-in scale',\n"
        "]"
    )

    pdf.sub_title("5.3 Customizing Labels for Your Domain")
    pdf.body_text("To adapt CLIP for different materials or defect types, modify labels in config.py or .env:")

    pdf.step(1, "Write descriptive OK labels for your product")
    pdf.code_block(
        "# Example: PCB inspection\n"
        "CLIP_LABELS_OK = [\n"
        "    'a photo of a clean circuit board',\n"
        "    'a photo of a properly soldered PCB',\n"
        "    'a photo of a normal electronic board',\n"
        "]"
    )

    pdf.step(2, "Write descriptive NG labels for each defect type")
    pdf.code_block(
        "CLIP_LABELS_NG = [\n"
        "    'a photo of a circuit board with missing components',\n"
        "    'a photo of a PCB with solder bridges',\n"
        "    'a photo of a burnt circuit board',\n"
        "    'a photo of a PCB with tombstoned components',\n"
        "]"
    )

    pdf.step(3, "Tune the threshold")
    pdf.code_block(
        "# Lower = more sensitive (more NG detections)\n"
        "# Higher = more conservative (fewer false alarms)\n"
        "CLIP_DEFECT_THRESHOLD = 0.5    # Default\n"
        "CLIP_DEFECT_THRESHOLD = 0.3    # More sensitive\n"
        "CLIP_DEFECT_THRESHOLD = 0.7    # More conservative"
    )

    pdf.tip(
        "Use specific, descriptive phrases. 'a photo of a scratched metal surface' works "
        "much better than just 'scratch'. CLIP understands natural language."
    )

    # ── Section 6: Fine-Tuning Tips ──
    pdf.new_page()
    pdf.section_title(6, "Fine-Tuning Tips & Best Practices")

    pdf.sub_title("6.1 If Model Misses Defects (Low Recall)")
    pdf.bullet("Lower YOLO_DEFECT_CONFIDENCE from 0.15 to 0.05-0.10")
    pdf.bullet("Lower CLIP_DEFECT_THRESHOLD from 0.5 to 0.3")
    pdf.bullet("Add more training images of the missed defect type")
    pdf.bullet("Use data augmentation: mosaic, mixup, flips, rotation")

    pdf.sub_title("6.2 If Model Has False Alarms (Low Precision)")
    pdf.bullet("Raise YOLO_DEFECT_CONFIDENCE from 0.15 to 0.25-0.35")
    pdf.bullet("Raise CLIP_DEFECT_THRESHOLD from 0.5 to 0.6-0.7")
    pdf.bullet("Add more OK/normal images to the training set")
    pdf.bullet("Ensure annotations are accurate (no mislabeled boxes)")

    pdf.sub_title("6.3 Training Data Guidelines")
    pdf.bullet("Minimum 100 images per class, 300+ recommended")
    pdf.bullet("Balance classes: similar counts for each defect type")
    pdf.bullet("Include edge cases: partial defects, small defects, multiple defects")
    pdf.bullet("Vary conditions: lighting, angle, scale, background")
    pdf.bullet("Use augmentation for small datasets (Ultralytics does this automatically)")

    pdf.sub_title("6.4 Evaluation Commands")
    pdf.code_block(
        "# Validate on test set\n"
        "model = YOLO('weights/best.pt')\n"
        "metrics = model.val(\n"
        "    data='datasets/NEU-DET/data.yaml',\n"
        "    imgsz=640,\n"
        "    device='mps',\n"
        ")\n"
        "\n"
        "print(f'mAP50:    {metrics.box.map50:.4f}')\n"
        "print(f'mAP50-95: {metrics.box.map:.4f}')\n"
        "\n"
        "# Export to ONNX for deployment\n"
        "model.export(format='onnx', imgsz=640)"
    )

    # ── Section 7: Deployment ──
    pdf.new_page()
    pdf.section_title(7, "Deploying Trained Models")

    pdf.step(1, "Copy weights to the correct location")
    pdf.code_block(
        "# YOLO defect model\n"
        "cp runs/detect/train/weights/best.pt weights/best.pt\n"
        "\n"
        "# Update config if path changed\n"
        "# YOLO_DEFECT_WEIGHTS_PATH=weights/best.pt"
    )

    pdf.step(2, "Update config.py with your CLIP labels")
    pdf.body_text("Edit backend/app/core/config.py with your custom OK/NG labels.")

    pdf.step(3, "Restart the backend")
    pdf.code_block(
        "cd backend\n"
        "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    )

    pdf.step(4, "Verify via health check")
    pdf.code_block(
        "curl http://localhost:8000/health\n"
        "# Should show:\n"
        "#   yolo_defect_loaded: true\n"
        "#   clip_loaded: true"
    )

    pdf.step(5, "Test with the frontend")
    pdf.body_text(
        "Open http://localhost:3000, select 'YOLO+CLIP' pipeline, "
        "and upload a test image to verify detection results."
    )

    output = "docs/YOLO_CLIP_Training_Guide.pdf"
    pdf.output(output)
    print(f"Generated: {output}")


# ═════════════════════════════════════════════════════════════════════
#  GUIDE 2: CNN + ResNet
# ═════════════════════════════════════════════════════════════════════

def generate_cnn_resnet_guide():
    pdf = GuidePDF(accent=GuidePDF.ACCENT2)
    pdf.alias_nb_pages()
    pdf.cover_page(
        "CNN + ResNet Pipeline",
        "Training & Fine-Tuning Guide",
        GuidePDF.ACCENT2,
    )

    # ── Section 1: Overview ──
    pdf.new_page()
    pdf.section_title(1, "Pipeline Overview")
    pdf.body_text(
        "The CNN+ResNet pipeline uses a ResNet convolutional neural network for "
        "binary image classification (OK vs NG). Unlike the YOLO+CLIP pipeline which "
        "detects individual defects with bounding boxes, this pipeline classifies the "
        "entire image as pass or fail."
    )

    pdf.sub_title("Pipeline Flow")
    pdf.body_text("Camera -> Image -> ResNet CNN -> Classify OK/NG -> Save to DB -> Dashboard")

    pdf.sub_title("Model Architecture")
    pdf.bullet("Backbone: ResNet18 / ResNet34 / ResNet50 (configurable)")
    pdf.bullet("Pretrained on ImageNet (1000 classes), fine-tuned for 2-class OK/NG")
    pdf.bullet("Input: 224x224 RGB image (auto-resized)")
    pdf.bullet("Output: Probability for each class [OK, NG]")
    pdf.bullet("Bonus: GradCAM heatmap visualization for explainability")

    pdf.sub_title("When to Use CNN+ResNet vs YOLO+CLIP")
    pdf.bullet("CNN+ResNet: Fast binary pass/fail, no defect localization needed")
    pdf.bullet("CNN+ResNet: Simpler to train, just need OK/NG image folders")
    pdf.bullet("YOLO+CLIP: Need defect bounding boxes and classification")
    pdf.bullet("YOLO+CLIP: Need to know what type of defect and where")

    # ── Section 2: Prerequisites ──
    pdf.new_page()
    pdf.section_title(2, "Prerequisites")

    pdf.sub_title("Hardware Requirements")
    pdf.bullet("GPU recommended: NVIDIA (CUDA) or Apple Silicon (MPS)")
    pdf.bullet("Minimum 8GB RAM")
    pdf.bullet("At least 2GB free disk space")

    pdf.sub_title("Software Requirements")
    pdf.code_block(
        "# Python 3.10+\n"
        "pip install torch>=2.1.0 torchvision>=0.16.0\n"
        "pip install opencv-python-headless\n"
        "pip install numpy<2.0.0 Pillow\n"
        "pip install scikit-learn  # for evaluation metrics"
    )

    pdf.sub_title("Project Structure")
    pdf.code_block(
        "vlm-yolo/\n"
        "  backend/\n"
        "    train_resnet.py              # Training script (create)\n"
        "    app/models/cnn_models.py     # ResNet classifier\n"
        "    app/core/config.py           # CNN config settings\n"
        "  weights/\n"
        "    resnet_classifier.pth        # Trained weights (output)"
    )

    # ── Section 3: Dataset Preparation ──
    pdf.new_page()
    pdf.section_title(3, "Dataset Preparation")

    pdf.body_text(
        "The CNN+ResNet pipeline requires a simple folder-based dataset. No bounding box "
        "annotations needed - just sort images into OK and NG folders."
    )

    pdf.sub_title("3.1 Required Directory Structure")
    pdf.code_block(
        "data/classification/\n"
        "  train/\n"
        "    OK/                # Good/passing images\n"
        "      img_001.jpg\n"
        "      img_002.jpg\n"
        "      ...\n"
        "    NG/                # Defective/failing images\n"
        "      img_001.jpg\n"
        "      img_002.jpg\n"
        "      ...\n"
        "  val/\n"
        "    OK/                # Validation good images\n"
        "      ...\n"
        "    NG/                # Validation defective images\n"
        "      ..."
    )

    pdf.step(1, "Collect images")
    pdf.bullet("Photograph parts on your production line or inspection station")
    pdf.bullet("Capture both OK (good) and NG (defective) samples")
    pdf.bullet("Aim for at least 100 images per class, 500+ for best results")
    pdf.bullet("Vary lighting, angle, and position for robustness")

    pdf.step(2, "Sort into folders")
    pdf.bullet("Create train/OK/ and train/NG/ directories")
    pdf.bullet("Create val/OK/ and val/NG/ directories")
    pdf.bullet("Split roughly 80% train / 20% validation")
    pdf.bullet("Ensure both splits have both classes represented")

    pdf.step(3, "Verify dataset balance")
    pdf.code_block(
        "import os\n"
        "for split in ['train', 'val']:\n"
        "    for cls in ['OK', 'NG']:\n"
        "        path = f'data/classification/{split}/{cls}'\n"
        "        count = len(os.listdir(path))\n"
        "        print(f'{split}/{cls}: {count} images')"
    )

    pdf.warning(
        "Imbalanced datasets (e.g. 1000 OK vs 50 NG) will bias the model. "
        "Use data augmentation or weighted loss to compensate."
    )

    # ── Section 4: Training Script ──
    pdf.new_page()
    pdf.section_title(4, "Training the ResNet Classifier")

    pdf.sub_title("4.1 Complete Training Script")
    pdf.body_text("Create backend/train_resnet.py with the following code:")

    pdf.code_block(
        '"""Train ResNet for binary OK/NG classification."""\n'
        "\n"
        "import argparse\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.data import DataLoader\n"
        "from torchvision import datasets, transforms, models\n"
        "from pathlib import Path"
    )

    pdf.code_block(
        "# Data transforms\n"
        "train_transform = transforms.Compose([\n"
        "    transforms.Resize(256),\n"
        "    transforms.RandomCrop(224),\n"
        "    transforms.RandomHorizontalFlip(),\n"
        "    transforms.RandomVerticalFlip(),\n"
        "    transforms.RandomRotation(15),\n"
        "    transforms.ColorJitter(\n"
        "        brightness=0.2, contrast=0.2, saturation=0.1),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize(\n"
        "        mean=[0.485, 0.456, 0.406],\n"
        "        std=[0.229, 0.224, 0.225]),\n"
        "])\n"
        "\n"
        "val_transform = transforms.Compose([\n"
        "    transforms.Resize(256),\n"
        "    transforms.CenterCrop(224),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize(\n"
        "        mean=[0.485, 0.456, 0.406],\n"
        "        std=[0.229, 0.224, 0.225]),\n"
        "])"
    )

    pdf.new_page()
    pdf.code_block(
        "def train(data_dir, epochs=30, batch_size=32,\n"
        "          lr=1e-3, arch='resnet18', device=None):\n"
        "\n"
        "    device = device or (\n"
        "        'cuda' if torch.cuda.is_available()\n"
        "        else 'mps' if torch.backends.mps.is_available()\n"
        "        else 'cpu')\n"
        "    print(f'Training on: {device}')\n"
        "\n"
        "    # Load datasets\n"
        "    train_ds = datasets.ImageFolder(\n"
        "        f'{data_dir}/train', train_transform)\n"
        "    val_ds = datasets.ImageFolder(\n"
        "        f'{data_dir}/val', val_transform)\n"
        "\n"
        "    print(f'Classes: {train_ds.classes}')\n"
        "    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')\n"
        "\n"
        "    train_loader = DataLoader(train_ds,\n"
        "        batch_size=batch_size, shuffle=True,\n"
        "        num_workers=2, pin_memory=True)\n"
        "    val_loader = DataLoader(val_ds,\n"
        "        batch_size=batch_size, shuffle=False,\n"
        "        num_workers=2, pin_memory=True)"
    )

    pdf.code_block(
        "    # Build model\n"
        "    arch_map = {\n"
        "        'resnet18': models.resnet18,\n"
        "        'resnet34': models.resnet34,\n"
        "        'resnet50': models.resnet50,\n"
        "    }\n"
        "    model = arch_map[arch](weights='IMAGENET1K_V1')\n"
        "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
        "    model = model.to(device)\n"
        "\n"
        "    criterion = nn.CrossEntropyLoss()\n"
        "    optimizer = torch.optim.Adam(\n"
        "        model.parameters(), lr=lr, weight_decay=1e-4)\n"
        "    scheduler = torch.optim.lr_scheduler.StepLR(\n"
        "        optimizer, step_size=10, gamma=0.1)"
    )

    pdf.new_page()
    pdf.code_block(
        "    best_acc = 0.0\n"
        "    for epoch in range(1, epochs + 1):\n"
        "        # --- Train ---\n"
        "        model.train()\n"
        "        running_loss, correct, total = 0, 0, 0\n"
        "        for images, labels in train_loader:\n"
        "            images = images.to(device)\n"
        "            labels = labels.to(device)\n"
        "            optimizer.zero_grad()\n"
        "            outputs = model(images)\n"
        "            loss = criterion(outputs, labels)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            running_loss += loss.item()\n"
        "            correct += (outputs.argmax(1) == labels).sum()\n"
        "            total += labels.size(0)\n"
        "        scheduler.step()\n"
        "        train_acc = correct / total * 100"
    )

    pdf.code_block(
        "        # --- Validate ---\n"
        "        model.eval()\n"
        "        val_correct, val_total = 0, 0\n"
        "        with torch.no_grad():\n"
        "            for images, labels in val_loader:\n"
        "                images = images.to(device)\n"
        "                labels = labels.to(device)\n"
        "                outputs = model(images)\n"
        "                val_correct += (\n"
        "                    outputs.argmax(1) == labels).sum()\n"
        "                val_total += labels.size(0)\n"
        "        val_acc = val_correct / val_total * 100\n"
        "\n"
        "        print(f'Epoch {epoch}/{epochs} '\n"
        "              f'Train: {train_acc:.1f}% '\n"
        "              f'Val: {val_acc:.1f}%')\n"
        "\n"
        "        if val_acc > best_acc:\n"
        "            best_acc = val_acc\n"
        "            Path('weights').mkdir(exist_ok=True)\n"
        "            torch.save(model.state_dict(),\n"
        "                       'weights/resnet_classifier.pth')\n"
        "            print(f'  Saved best model ({val_acc:.1f}%)')\n"
        "\n"
        "    print(f'Training complete. Best: {best_acc:.1f}%')"
    )

    pdf.sub_title("4.2 Run Training")
    pdf.code_block(
        "cd backend\n"
        "python train_resnet.py \\\n"
        "    --data-dir ../data/classification \\\n"
        "    --epochs 30 \\\n"
        "    --batch-size 32 \\\n"
        "    --lr 0.001 \\\n"
        "    --arch resnet18"
    )

    # ── Section 5: Step-by-Step Training ──
    pdf.new_page()
    pdf.section_title(5, "Step-by-Step Training Walkthrough")

    pdf.step(1, "Prepare your dataset (see Section 3)")
    pdf.body_text("Sort images into data/classification/train/OK, train/NG, val/OK, val/NG")

    pdf.step(2, "Choose your backbone architecture")
    pdf.bullet("ResNet18 (11.7M params) - Fastest, good for small datasets")
    pdf.bullet("ResNet34 (21.8M params) - Balanced speed/accuracy")
    pdf.bullet("ResNet50 (25.6M params) - Best accuracy, needs more data")
    pdf.tip("Start with ResNet18. Only upgrade if accuracy plateaus.")

    pdf.step(3, "Set hyperparameters")
    pdf.bullet("Learning rate: Start with 0.001, reduce if loss oscillates")
    pdf.bullet("Batch size: 32 (reduce to 16 if out of memory)")
    pdf.bullet("Epochs: 30 (increase if val accuracy still improving)")

    pdf.step(4, "Run training and monitor metrics")
    pdf.body_text("Watch for these patterns:")
    pdf.bullet("Train acc increases, val acc increases -> Good, keep training")
    pdf.bullet("Train acc increases, val acc plateaus -> Start of overfitting")
    pdf.bullet("Train acc increases, val acc decreases -> Overfitting! Stop or add regularization")

    pdf.step(5, "Evaluate the final model")
    pdf.code_block(
        "# Load and test\n"
        "import torch\n"
        "from torchvision import models\n"
        "\n"
        "model = models.resnet18(weights=None)\n"
        "model.fc = torch.nn.Linear(512, 2)\n"
        "model.load_state_dict(\n"
        "    torch.load('weights/resnet_classifier.pth'))\n"
        "model.eval()\n"
        "\n"
        "# Compute accuracy, precision, recall, F1\n"
        "from sklearn.metrics import classification_report\n"
        "# ... run inference on val set ..."
    )

    # ── Section 6: Fine-Tuning Strategies ──
    pdf.new_page()
    pdf.section_title(6, "Fine-Tuning Strategies")

    pdf.sub_title("6.1 Transfer Learning (Recommended)")
    pdf.body_text(
        "The default approach: load ImageNet-pretrained weights, replace the final layer, "
        "and train the entire network. Works well with 200+ images per class."
    )

    pdf.sub_title("6.2 Feature Extraction (Small Datasets)")
    pdf.body_text("Freeze the backbone and only train the classifier head. Best for < 200 images:")
    pdf.code_block(
        "# Freeze all backbone layers\n"
        "for param in model.parameters():\n"
        "    param.requires_grad = False\n"
        "\n"
        "# Unfreeze only the classifier head\n"
        "model.fc = nn.Linear(model.fc.in_features, 2)\n"
        "for param in model.fc.parameters():\n"
        "    param.requires_grad = True\n"
        "\n"
        "# Use higher learning rate since only\n"
        "# training last layer\n"
        "optimizer = Adam(model.fc.parameters(), lr=0.01)"
    )

    pdf.sub_title("6.3 Progressive Unfreezing (Advanced)")
    pdf.body_text("Start frozen, gradually unfreeze deeper layers:")
    pdf.code_block(
        "# Phase 1: Train head only (5 epochs, lr=0.01)\n"
        "# Phase 2: Unfreeze layer4 (5 epochs, lr=0.001)\n"
        "# Phase 3: Unfreeze all (10 epochs, lr=0.0001)"
    )

    pdf.sub_title("6.4 Handling Imbalanced Data")
    pdf.code_block(
        "# Option A: Weighted loss\n"
        "# If 80% OK, 20% NG:\n"
        "weights = torch.tensor([0.2, 0.8])  # [OK, NG]\n"
        "criterion = nn.CrossEntropyLoss(\n"
        "    weight=weights.to(device))\n"
        "\n"
        "# Option B: Oversampling\n"
        "from torch.utils.data import WeightedRandomSampler\n"
        "# ... create sampler with higher weight for\n"
        "# minority class ..."
    )

    # ── Section 7: Deployment ──
    pdf.new_page()
    pdf.section_title(7, "Deploying the Trained Model")

    pdf.step(1, "Verify weights file exists")
    pdf.code_block(
        "ls -la weights/resnet_classifier.pth\n"
        "# Should show your trained model file"
    )

    pdf.step(2, "Update config.py if needed")
    pdf.code_block(
        "# In backend/app/core/config.py\n"
        "CNN_RESNET_WEIGHTS_PATH = 'weights/resnet_classifier.pth'\n"
        "CNN_RESNET_ARCH = 'resnet18'       # must match training\n"
        "CNN_RESNET_NUM_CLASSES = 2\n"
        "CNN_RESNET_CLASS_NAMES = ['OK', 'NG']\n"
        "CNN_RESNET_THRESHOLD = 0.5"
    )

    pdf.warning(
        "The class order must match training! PyTorch ImageFolder sorts alphabetically: "
        "NG=0, OK=1. The config CNN_RESNET_CLASS_NAMES=['OK','NG'] maps index 0->OK, 1->NG. "
        "Verify your folder ordering matches."
    )

    pdf.step(3, "Restart the backend")
    pdf.code_block(
        "cd backend\n"
        "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    )

    pdf.step(4, "Verify via health check")
    pdf.code_block(
        "curl http://localhost:8000/health\n"
        "# Should show: resnet_loaded: true"
    )

    pdf.step(5, "Test with the frontend")
    pdf.body_text(
        "Open http://localhost:3000, switch to 'CNN+ResNet' pipeline, "
        "and upload a test image. The result will show OK or NG with confidence."
    )

    pdf.sub_title("7.1 Tuning the Classification Threshold")
    pdf.code_block(
        "# In config.py or .env:\n"
        "CNN_RESNET_THRESHOLD = 0.5   # Default\n"
        "CNN_RESNET_THRESHOLD = 0.3   # More sensitive (catch more NG)\n"
        "CNN_RESNET_THRESHOLD = 0.7   # More conservative (fewer false NG)"
    )
    pdf.body_text(
        "Adjust based on your quality requirements: lower threshold catches more defects "
        "but may increase false alarms. Higher threshold reduces false alarms but may miss defects."
    )

    # ── Section 8: GradCAM ──
    pdf.new_page()
    pdf.section_title(8, "GradCAM Heatmap Visualization")

    pdf.body_text(
        "The CNN+ResNet pipeline includes GradCAM (Gradient-weighted Class Activation Mapping) "
        "to visualize which image regions influenced the model's decision. This helps debug "
        "false predictions and build trust in the model."
    )

    pdf.sub_title("How It Works")
    pdf.bullet("1. Forward pass through the network to get prediction")
    pdf.bullet("2. Backward pass from predicted class through last conv layer")
    pdf.bullet("3. Compute weighted activation maps based on gradients")
    pdf.bullet("4. Overlay heatmap on original image (red = high attention)")

    pdf.sub_title("Using GradCAM via API")
    pdf.code_block(
        "# After running a CNN inspection:\n"
        "GET /cnn/inspections/{inspection_id}/gradcam\n"
        "\n"
        "# Returns a PNG heatmap image\n"
        "# Red regions = areas the model focused on\n"
        "# Blue regions = areas the model ignored"
    )

    pdf.tip(
        "If GradCAM shows the model focusing on the wrong areas (background, edges), "
        "your training data may need better variety or the model is overfitting to artifacts."
    )

    output = "docs/CNN_ResNet_Training_Guide.pdf"
    pdf.output(output)
    print(f"Generated: {output}")


# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    generate_yolo_clip_guide()
    generate_cnn_resnet_guide()
    print("\nDone! Both guides generated in docs/")
