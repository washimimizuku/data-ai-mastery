# Project D: Computer Vision Pipeline

## Objective

Build a comprehensive computer vision system using state-of-the-art open-source models for object detection, segmentation, classification, OCR, and image captioning with a unified Gradio interface.

**What You'll Build**: A production-ready CV pipeline that performs 5 different vision tasks (YOLO object detection, SAM segmentation, ViT classification, TrOCR text extraction, BLIP-2 captioning) with batch processing, performance benchmarks, and an interactive web UI.

**What You'll Learn**: Modern computer vision techniques, Hugging Face Transformers, YOLO architecture, Segment Anything Model, Vision Transformers, OCR systems, image captioning, and building unified CV applications.

## Time Estimate

**1-2 days (8-16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hours 1-2**: Setup and install models (PyTorch, transformers, ultralytics, SAM)
- **Hours 3-4**: Implement object detection (YOLO v8/v10, test on images)
- **Hours 5-6**: Implement segmentation (SAM, mask generation)
- **Hours 7-8**: Implement classification (ViT, test on ImageNet classes)

### Day 2 (8 hours) - If needed
- **Hours 1-2**: Implement OCR (TrOCR/EasyOCR, text extraction)
- **Hours 3-4**: Implement captioning (BLIP-2, generate descriptions)
- **Hours 5-6**: Build Gradio interface (all tasks in one UI, batch processing)
- **Hour 7**: Benchmark performance (speed, accuracy metrics)
- **Hour 8**: Documentation and polish (README, examples, cleanup)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 51-70
  - Days 51-60: ML fundamentals
  - Days 61-70: Deep learning basics
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM (16GB recommended)
- GPU with 4GB+ VRAM (optional, for faster inference)
- Understanding of CNNs and transformers
- Basic PyTorch knowledge

### Tools Needed
- Python with torch, transformers, ultralytics
- OpenCV for image preprocessing
- Gradio for UI
- Sample images for testing
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision transformers

# Install CV libraries
pip install ultralytics opencv-python pillow

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install EasyOCR (alternative to TrOCR)
pip install easyocr

# Install Gradio
pip install gradio

# Create project structure
mkdir -p cv-pipeline/{src,data/images,results}
cd cv-pipeline
```

### Step 3: Implement Object Detection (YOLO)
```python
# src/detection.py
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

class ObjectDetector:
    """Object detection using YOLO v8"""
    
    def __init__(self, model_size: str = "yolov8n"):
        """
        Initialize YOLO model
        Args:
            model_size: n (nano), s (small), m (medium), l (large), x (xlarge)
        """
        self.model = YOLO(f"{model_size}.pt")
        print(f"✓ Loaded {model_size}")
    
    def detect(self, image_path: str, conf_threshold: float = 0.25):
        """
        Detect objects in image
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold (0-1)
        Returns:
            dict with detections and annotated image
        """
        # Run inference
        results = self.model(image_path, conf=conf_threshold)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        
        # Get annotated image
        annotated = results[0].plot()
        annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR to RGB
        
        return {
            "detections": detections,
            "count": len(detections),
            "annotated_image": annotated_pil
        }

# Usage
if __name__ == "__main__":
    detector = ObjectDetector("yolov8n")
    result = detector.detect("data/images/sample.jpg")
    
    print(f"Found {result['count']} objects:")
    for det in result['detections']:
        print(f"  - {det['class']}: {det['confidence']:.2f}")
    
    result['annotated_image'].save("results/detection.jpg")
```

### Step 4: Implement Segmentation (SAM)
```python
# src/segmentation.py
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
from PIL import Image

class ImageSegmenter:
    """Image segmentation using Segment Anything Model"""
    
    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = "sam_vit_b.pth"):
        """
        Initialize SAM
        Args:
            model_type: vit_b, vit_l, or vit_h
            checkpoint_path: Path to SAM checkpoint
        """
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        print(f"✓ Loaded SAM {model_type}")
    
    def segment(self, image_path: str):
        """
        Generate segmentation masks
        Args:
            image_path: Path to image
        Returns:
            dict with masks and visualization
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = self.mask_generator.generate(image_rgb)
        
        # Create visualization
        vis = self._visualize_masks(image_rgb, masks)
        
        return {
            "num_masks": len(masks),
            "masks": masks,
            "visualization": Image.fromarray(vis)
        }
    
    def _visualize_masks(self, image, masks):
        """Create colored mask overlay"""
        overlay = image.copy()
        
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, 3).tolist()
            overlay[mask['segmentation']] = color
        
        # Blend with original
        result = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        return result

# Usage
if __name__ == "__main__":
    segmenter = ImageSegmenter()
    result = segmenter.segment("data/images/sample.jpg")
    
    print(f"Generated {result['num_masks']} masks")
    result['visualization'].save("results/segmentation.jpg")
```

### Step 5: Implement Classification (ViT)
```python
# src/classification.py
from transformers import pipeline
from PIL import Image

class ImageClassifier:
    """Image classification using Vision Transformer"""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize ViT classifier
        Args:
            model_name: Hugging Face model name
        """
        self.classifier = pipeline(
            "image-classification",
            model=model_name
        )
        print(f"✓ Loaded {model_name}")
    
    def classify(self, image_path: str, top_k: int = 5):
        """
        Classify image
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
        Returns:
            list of predictions with labels and scores
        """
        image = Image.open(image_path)
        
        # Get predictions
        predictions = self.classifier(image, top_k=top_k)
        
        return {
            "predictions": predictions,
            "top_class": predictions[0]["label"],
            "confidence": predictions[0]["score"]
        }

# Usage
if __name__ == "__main__":
    classifier = ImageClassifier()
    result = classifier.classify("data/images/sample.jpg")
    
    print(f"Top prediction: {result['top_class']} ({result['confidence']:.2%})")
    print("\nAll predictions:")
    for pred in result['predictions']:
        print(f"  {pred['label']}: {pred['score']:.2%}")
```

### Step 6: Implement OCR (TrOCR)
```python
# src/ocr.py
from transformers import pipeline
from PIL import Image
import easyocr

class TextExtractor:
    """OCR using TrOCR and EasyOCR"""
    
    def __init__(self, use_trocr: bool = True):
        """
        Initialize OCR model
        Args:
            use_trocr: Use TrOCR (True) or EasyOCR (False)
        """
        self.use_trocr = use_trocr
        
        if use_trocr:
            self.ocr = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-printed"
            )
            print("✓ Loaded TrOCR")
        else:
            self.ocr = easyocr.Reader(['en'])
            print("✓ Loaded EasyOCR")
    
    def extract_text(self, image_path: str):
        """
        Extract text from image
        Args:
            image_path: Path to image
        Returns:
            dict with extracted text
        """
        if self.use_trocr:
            image = Image.open(image_path)
            result = self.ocr(image)
            text = result[0]["generated_text"]
        else:
            result = self.ocr.readtext(image_path)
            text = " ".join([detection[1] for detection in result])
        
        return {
            "text": text,
            "length": len(text),
            "words": len(text.split())
        }

# Usage
if __name__ == "__main__":
    extractor = TextExtractor(use_trocr=True)
    result = extractor.extract_text("data/images/text_image.jpg")
    
    print(f"Extracted text ({result['words']} words):")
    print(result['text'])
```

### Step 7: Implement Image Captioning (BLIP-2)
```python
# src/captioning.py
from transformers import pipeline
from PIL import Image

class ImageCaptioner:
    """Image captioning using BLIP-2"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize BLIP-2 model
        Args:
            model_name: Hugging Face model name
        """
        self.captioner = pipeline(
            "image-to-text",
            model=model_name
        )
        print(f"✓ Loaded {model_name}")
    
    def caption(self, image_path: str, max_length: int = 50):
        """
        Generate image caption
        Args:
            image_path: Path to image
            max_length: Maximum caption length
        Returns:
            dict with caption
        """
        image = Image.open(image_path)
        
        # Generate caption
        result = self.captioner(image, max_new_tokens=max_length)
        caption = result[0]["generated_text"]
        
        return {
            "caption": caption,
            "length": len(caption)
        }

# Usage
if __name__ == "__main__":
    captioner = ImageCaptioner()
    result = captioner.caption("data/images/sample.jpg")
    
    print(f"Caption: {result['caption']}")
```

### Step 8: Build Unified Gradio Interface
```python
# app.py
import gradio as gr
from src.detection import ObjectDetector
from src.segmentation import ImageSegmenter
from src.classification import ImageClassifier
from src.ocr import TextExtractor
from src.captioning import ImageCaptioner

# Initialize all models
print("Loading models...")
detector = ObjectDetector("yolov8n")
# segmenter = ImageSegmenter()  # Requires SAM checkpoint
classifier = ImageClassifier()
ocr = TextExtractor(use_trocr=False)  # EasyOCR is easier to setup
captioner = ImageCaptioner()
print("✓ All models loaded!")

def detect_objects(image, conf_threshold):
    """Object detection tab"""
    result = detector.detect(image, conf_threshold)
    
    output = f"**Found {result['count']} objects:**\n\n"
    for det in result['detections']:
        output += f"- {det['class']}: {det['confidence']:.2%}\n"
    
    return result['annotated_image'], output

def classify_image(image, top_k):
    """Classification tab"""
    result = classifier.classify(image, top_k)
    
    output = f"**Top prediction:** {result['top_class']} ({result['confidence']:.2%})\n\n"
    output += "**All predictions:**\n"
    for pred in result['predictions']:
        output += f"- {pred['label']}: {pred['score']:.2%}\n"
    
    return output

def extract_text(image):
    """OCR tab"""
    result = ocr.extract_text(image)
    
    output = f"**Extracted {result['words']} words:**\n\n{result['text']}"
    return output

def generate_caption(image):
    """Captioning tab"""
    result = captioner.caption(image)
    return result['caption']

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Computer Vision Pipeline") as demo:
    gr.Markdown("# 🖼️ Computer Vision Pipeline")
    gr.Markdown("Perform multiple CV tasks: object detection, classification, OCR, and captioning")
    
    with gr.Tabs():
        # Object Detection Tab
        with gr.Tab("Object Detection"):
            with gr.Row():
                with gr.Column():
                    det_input = gr.Image(type="filepath", label="Upload Image")
                    det_conf = gr.Slider(0.1, 0.9, value=0.25, label="Confidence Threshold")
                    det_btn = gr.Button("Detect Objects", variant="primary")
                
                with gr.Column():
                    det_output_img = gr.Image(label="Detections")
                    det_output_text = gr.Markdown()
            
            det_btn.click(
                fn=detect_objects,
                inputs=[det_input, det_conf],
                outputs=[det_output_img, det_output_text]
            )
        
        # Classification Tab
        with gr.Tab("Classification"):
            with gr.Row():
                with gr.Column():
                    cls_input = gr.Image(type="filepath", label="Upload Image")
                    cls_topk = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    cls_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column():
                    cls_output = gr.Markdown()
            
            cls_btn.click(
                fn=classify_image,
                inputs=[cls_input, cls_topk],
                outputs=cls_output
            )
        
        # OCR Tab
        with gr.Tab("OCR"):
            with gr.Row():
                with gr.Column():
                    ocr_input = gr.Image(type="filepath", label="Upload Image with Text")
                    ocr_btn = gr.Button("Extract Text", variant="primary")
                
                with gr.Column():
                    ocr_output = gr.Markdown()
            
            ocr_btn.click(
                fn=extract_text,
                inputs=ocr_input,
                outputs=ocr_output
            )
        
        # Captioning Tab
        with gr.Tab("Image Captioning"):
            with gr.Row():
                with gr.Column():
                    cap_input = gr.Image(type="filepath", label="Upload Image")
                    cap_btn = gr.Button("Generate Caption", variant="primary")
                
                with gr.Column():
                    cap_output = gr.Textbox(label="Caption", lines=3)
            
            cap_btn.click(
                fn=generate_caption,
                inputs=cap_input,
                outputs=cap_output
            )

if __name__ == "__main__":
    demo.launch(share=False)
```

## Key Features to Implement

### 1. Object Detection (YOLO v8/v10)
- Detect multiple objects in images
- Bounding box visualization
- Confidence thresholding
- Support for 80+ COCO classes

### 2. Image Segmentation (SAM)
- Automatic mask generation
- Instance segmentation
- Colored mask overlay
- Mask export

### 3. Image Classification (ViT)
- ImageNet classification (1000 classes)
- Top-k predictions
- Confidence scores
- Vision Transformer architecture

### 4. OCR (TrOCR/EasyOCR)
- Text extraction from images
- Printed and handwritten text
- Multiple language support
- Bounding box detection

### 5. Image Captioning (BLIP-2)
- Natural language descriptions
- Context-aware captions
- Adjustable caption length

### 6. Gradio Interface
- All tasks in one UI
- Tabbed interface
- Image upload
- Real-time processing

### 7. Batch Processing
- Process multiple images
- Export results
- Performance tracking

### 8. Performance Benchmarks
- Inference speed (FPS)
- Model size
- Accuracy metrics
- GPU vs CPU comparison

## Success Criteria

By the end of this project, you should have:

### Functionality
- [ ] Object detection working (YOLO v8)
- [ ] Segmentation working (SAM)
- [ ] Classification working (ViT)
- [ ] OCR working (TrOCR or EasyOCR)
- [ ] Captioning working (BLIP-2)
- [ ] Gradio interface with all tasks
- [ ] Batch processing capability

### Quality Metrics
- [ ] **All CV tasks working**: 5/5 tasks functional
- [ ] **Interactive UI functional**: Gradio app running
- [ ] **Performance benchmarks documented**: Speed and accuracy
- [ ] **Code quality**: < 600 lines of code
- [ ] **Inference speed**: < 2 seconds per image (GPU)

### Deliverables
- [ ] 5 CV task implementations
- [ ] Unified Gradio interface
- [ ] Performance benchmark results
- [ ] Sample images and outputs
- [ ] Comprehensive documentation

## Learning Outcomes

After completing this project, you'll be able to:

- Understand modern CV architectures (YOLO, ViT, SAM, BLIP)
- Use Hugging Face Transformers for vision tasks
- Implement object detection with YOLO
- Perform image segmentation with SAM
- Build classification systems with ViT
- Extract text with OCR models
- Generate image captions
- Create unified CV pipelines
- Build interactive CV demos with Gradio

## Expected Performance

**Inference Speed (GPU)**:
```
Object Detection (YOLO v8n): 20-30 FPS
Segmentation (SAM): 1-2 seconds
Classification (ViT): 50-100 FPS
OCR (EasyOCR): 1-3 seconds
Captioning (BLIP-2): 2-4 seconds
```

**Model Sizes**:
```
YOLO v8n: 6 MB
SAM ViT-B: 375 MB
ViT-Base: 330 MB
TrOCR: 1.4 GB
BLIP-2: 2.7 GB
```

**Accuracy**:
```
YOLO v8n mAP: 37.3% (COCO)
ViT-Base Top-1: 81.8% (ImageNet)
BLIP-2: State-of-the-art captioning
```

## Project Structure

```
project-d-computer-vision/
├── src/
│   ├── __init__.py
│   ├── detection.py         # YOLO object detection
│   ├── segmentation.py      # SAM segmentation
│   ├── classification.py    # ViT classification
│   ├── ocr.py               # TrOCR/EasyOCR text extraction
│   ├── captioning.py        # BLIP-2 captioning
│   └── utils.py             # Helper functions
├── data/
│   └── images/              # Sample images
├── results/
│   ├── detection/           # Detection outputs
│   ├── segmentation/        # Segmentation masks
│   └── benchmarks.json      # Performance metrics
├── tests/
│   ├── test_detection.py
│   ├── test_classification.py
│   └── test_ocr.py
├── notebooks/
│   └── cv_pipeline_demo.ipynb
├── app.py                   # Gradio UI
├── benchmark.py             # Performance testing
├── requirements.txt
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Model Download Size
**Problem**: Models are large (2-3 GB total)
**Solution**: Download on-demand, use smaller variants (YOLO nano, ViT base)

### Challenge 2: GPU Memory
**Problem**: Running all models simultaneously
**Solution**: Load models on-demand, use CPU for some tasks

### Challenge 3: SAM Setup
**Problem**: SAM requires checkpoint download
**Solution**: Use alternative segmentation or provide download script

### Challenge 4: Slow Inference
**Problem**: Processing takes too long
**Solution**: Use GPU, optimize batch size, use smaller models

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with sample outputs
2. **Write Blog Post**: "Building a Unified Computer Vision Pipeline"
3. **Extend Features**: Add video processing, real-time webcam
4. **Build Project E**: Continue with NLP Multi-Task System
5. **Production Use**: Deploy with FastAPI backend

## Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Segment Anything](https://segment-anything.com/)
- [Hugging Face Vision](https://huggingface.co/models?pipeline_tag=computer-vision)
- [PyTorch Vision](https://pytorch.org/vision/)
- [Gradio Documentation](https://gradio.app/)

## Questions?

If you get stuck:
1. Review the tech-spec.md for model details
2. Check implementation-plan.md for step-by-step guide
3. Search Hugging Face forums for model-specific issues
4. Review the 100 Days bootcamp materials on computer vision
