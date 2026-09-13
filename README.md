# Is Depth Worth the Latency?

## A Lightweight, Jointly-Trained Dehaze-and-Detect Kernel for Laparoscopic Video

This notebook implements an **edge-oriented computer vision pipeline for surgical smoke removal and downstream perception in laparoscopic video**.

The project investigates a practical deployment question:

> **If an additional sensing modality improves accuracy, does that improvement justify the true end-to-end cost of obtaining that modality?**

We study this question through lightweight image dehazing, RGB+depth fusion, and jointly trained surgical instrument/organ detection.

---

## Project Overview

Electrocautery during laparoscopic surgery produces smoke that can significantly degrade visibility and obscure anatomical structures and surgical instruments.

While image dehazing has been extensively studied for outdoor scenes, surgical smoke presents a different setting:

* Very short scene-depth ranges
* Artificial and close-range illumination
* Rapidly changing smoke density
* Highly constrained deployment environments

This project therefore focuses not only on image restoration quality, but also on **model size, inference latency, sensor-fusion overhead, quantization behavior, and downstream detection performance**.

##  What This Notebook Investigates

The notebook contains four major experiments:

### 1. Classical Dehazing — Dark Channel Prior

A training-free **Dark Channel Prior (DCP)** implementation is used as a classical baseline.

We evaluate its:

* PSNR
* SSIM
* CPU latency
* Robustness across different synthetic smoke severities


### 2. Lightweight Learned Dehazing

A compact AOD-Net-inspired convolutional network is trained for surgical image dehazing.

Two configurations are evaluated:

| Model           | Input                 | Parameters |
| --------------- | --------------------- | ---------: |
| RGB-only CNN    | RGB                   |      1,761 |
| RGB + Depth CNN | RGB + monocular depth |      1,764 |

The extremely small parameter count makes the model suitable for investigating **resource-constrained edge inference**.


### 3. RGB + Depth Sensor Fusion

Monocular depth estimated using **MiDaS** is introduced as an auxiliary input.

The important distinction is that we measure the **complete cost of obtaining the depth signal**, rather than treating depth as a free input.

This allows us to compare:

**Accuracy improvement vs. end-to-end latency cost**

The depth-fusion experiments are evaluated across **seven independent trials** to account for training and evaluation variability.


### 4. Joint Dehaze-and-Detect Kernel

The dehazing network is extended into a jointly trained perception model.

Instead of:

```text
Hazy Image
     ↓
Dehazing Network
     ↓
Clean Image
     ↓
Detector
     ↓
Detections
```

the proposed architecture shares computation:

```text
             ┌───────────────┐
Hazy Frame → │ Dehazing Stem │
             └───────┬───────┘
                     ↓
             Shared Backbone
                     ↓
             Detection Head
                     ↓
              Detections
```

The model is trained end-to-end using a combined dehazing and detection objective.

## Dataset

The experiments use **CholecSeg8k**, a laparoscopic cholecystectomy dataset containing pixel-level semantic segmentation masks.

The dataset contains:

* 8,080 annotated frames
* 101 laparoscopic video segments
* 13 semantic classes

For detection, existing segmentation masks are automatically converted into bounding-box annotations for:

* **Grasper**
* **Gallbladder**

No additional manual bounding-box annotation is required.

The resulting detection dataset contains:

* 4,556 annotated frames
* 4,702 gallbladder instances
* 244 grasper instances

---

## Synthetic Surgical Haze

Because CholecSeg8k does not provide paired clean/smoke-corrupted images, synthetic haze is generated using a physically motivated atmospheric scattering model:

```text
I_hazy(x) = I_clean(x) · t(x) + A · (1 − t(x))
```

where:

* `I_clean` = clean reference image
* `I_hazy` = generated hazy image
* `t(x)` = transmission map
* `A` = atmospheric light

A spatially varying random depth-proxy field is used to generate the transmission map.

Three smoke-severity levels are evaluated:

```text
β = 0.3  → Low
β = 0.6  → Medium
β = 0.9  → High
```

Training samples use randomly selected severity values in:

```text
β ∈ [0.2, 0.9]
```

A field-of-view mask is applied so that the circular endoscopic vignette does not artificially influence haze generation or evaluation metrics.

---

## Model Architecture

### Lightweight Dehazing Network

The learned dehazing model follows the AOD-Net formulation.

It consists of a compact five-layer convolutional stack with dense skip connections.

The network estimates a per-pixel parameter map `K`, from which the restored image is reconstructed.

Two input configurations are evaluated:

```text
RGB
 ↓
5-layer CNN
 ↓
Dehazed Image
```

and

```text
RGB + Depth
 ↓
5-layer CNN
 ↓
Dehazed Image
```

The depth channel acts only as auxiliary information for estimating the dehazing parameters.

---

## Joint Detection Architecture

The joint model extends the lightweight dehazing network with:

* A shared convolutional backbone
* Three stride-2 convolutional blocks
* A single-scale detection head
* Three anchor boxes per grid cell
* Objectness prediction
* Bounding-box regression
* Class prediction
* Thresholding + Non-Maximum Suppression

The network is deliberately kept simple to prioritize **edge-deployment latency**.

---

## Automatic Detection Annotation

Existing segmentation masks are converted into detection annotations automatically.

The pipeline:

1. Extracts the target semantic class
2. Identifies connected components
3. Removes components below a 50-pixel threshold
4. Computes tight axis-aligned bounding boxes
5. Produces detector-ready annotations

This avoids manual bounding-box labeling and can potentially be extended to the remaining CholecSeg8k classes.

---

## Training

### Dehazing

* Optimizer: Adam
* Initial learning rate: `1e-3`
* Training duration: 30 epochs
* Cosine annealing
* Random 256×256 crops/resizes
* FOV-masked L1 + SSIM loss

### Joint Dehaze + Detection

* Optimizer: Adam
* Learning rate: `3e-4`
* Gradient clipping
* Early stopping
* Class-weighted classification loss
* Synthetic hazy inputs
* Joint reconstruction + detection objective

The detection dataset uses class-stratified partitioning to ensure that the rarer grasper class is represented in both training and validation data.


## Edge Deployment Evaluation

Models are exported to **ONNX** and evaluated using CPU inference.

The notebook measures:

* Parameter count
* FP32 inference latency
* INT8 dynamic-quantization latency
* Stage-by-stage pipeline latency
* End-to-end latency

The CPU measurements are intended as a **software-based edge-deployment study** rather than a substitute for measurements on a dedicated embedded accelerator.

---

# Key Results

### Dehazing

At medium synthetic smoke severity (`β = 0.6`):

| Method              |           PSNR |        SSIM |     Latency |
| ------------------- | -------------: | ----------: | ----------: |
| DCP                 |       19.07 dB |       0.838 | ~100–107 ms |
| RGB CNN             | 19.37–19.93 dB | 0.808–0.825 |   ~10–13 ms |
| RGB + Depth CNN     | 19.76–20.46 dB | 0.807–0.852 |   ~11–13 ms |
| RGB + Depth + MiDaS | 19.76–20.46 dB | 0.807–0.852 | ~133–157 ms |

The fusion network itself adds very little latency.

The major computational cost comes from **producing the monocular depth estimate**.

## Depth-Fusion Finding

Across seven independent trials:

```text
Mean PSNR improvement: +0.224 dB
Standard deviation:    ±0.286 dB
Positive trials:       5 / 7
```

This suggests that depth provides a **modest but potentially real accuracy benefit** at this model scale.

However, obtaining the depth signal using MiDaS introduces approximately **122–145 ms** of additional computation.

Therefore:

> A small accuracy improvement can become difficult to justify when the complete sensor-fusion pipeline is considered.

This is one of the central deployment-oriented findings of the project.

## Quantization Finding

An additional experiment evaluates FP32 against INT8 dynamic quantization.

At this extremely small model scale, INT8 quantization does **not** provide the expected latency improvement.

Instead, measured latency increases substantially while PSNR remains approximately unchanged.

This demonstrates an important edge-optimization principle:

> **Quantization should be empirically validated on the target workload rather than assumed to reduce latency.**

---

## Joint Detection Result

The jointly trained dehaze-and-detect kernel achieves:

| Class       |    AP@0.5 |
| ----------- | --------: |
| Grasper     |     0.397 |
| Gallbladder |     0.078 |
| **mAP@0.5** | **0.237** |

The relatively low gallbladder AP indicates substantial room for improvement, particularly through better detection architecture, annotation quality, and multi-scale feature processing.

The current implementation intentionally prioritizes architectural simplicity and low additional latency.

---

## End-to-End Kernel Latency

Measured CPU latency:

| Stage                     | Approx. latency |
| ------------------------- | --------------: |
| Dehazing stem             |        ~15.2 ms |
| Backbone + detection head |         ~4.1 ms |
| **Full kernel**           |    **~17.7 ms** |

The dehazing stem accounts for the majority of the computation, while the detection capability adds comparatively little additional latency once shared features are available.

---

# Reproducibility

This notebook is designed as a reproducible experimental pipeline.

It includes:

* Synthetic haze generation
* FOV-aware masking
* DCP baseline
* Lightweight RGB dehazing
* RGB + depth fusion
* Automatic segmentation-to-detection annotation
* Joint dehaze-and-detect training
* Multi-run evaluation
* ONNX export
* CPU latency profiling
* Quantization experiments

---

# Limitations

The current study has several important limitations:

### Synthetic rather than real smoke

The haze model approximates surgical smoke but does not reproduce its full turbulent and non-uniform behavior.

### Monocular depth proxy

MiDaS is used as a proxy for a true depth-sensing modality. The study therefore measures the computational cost of generating depth-like information rather than validating an actual depth sensor.

### CPU-only deployment study

Latency measurements are obtained on general-purpose CPU hardware and should not be interpreted as measurements for a specific embedded SoC, DSP, or NPU.

### Limited detection scope

Only two classes are currently used:

* Grasper
* Gallbladder

### Dataset scope

The evaluation is limited to laparoscopic cholecystectomy imagery and does not establish generalization to other surgical procedures, cameras, hospitals, or imaging systems.

---

#  Future Work

The project can be extended in several directions:

* Validate against **real surgical smoke**
* Evaluate on the **ITEC Smoke Cholec80** dataset
* Deploy on an actual **embedded/edge accelerator**
* Investigate hardware-aware INT8 quantization
* Explore pruning and knowledge distillation
* Add temporal consistency for video
* Extend detection to additional CholecSeg8k classes
* Improve gallbladder detection with multi-scale features
* Evaluate alternative lightweight detectors
* Investigate hardware-specific acceleration using GPU/DSP/NPU backends
* Compare real depth sensors or stereo endoscopy against monocular depth estimation

---

# Central Takeaway

This project is not simply about removing smoke from surgical images.

It investigates a broader Edge AI question:

> **How much accuracy should we pay for, and what is the true computational cost of obtaining it?**

The results demonstrate that:

1. Lightweight learned dehazing can outperform classical DCP under heavier synthetic smoke.
2. Depth fusion can provide a modest accuracy improvement.
3. The cost of generating the auxiliary depth signal can dominate the actual fusion cost.
4. Extremely small networks can behave unexpectedly under conventional quantization.
5. Dehazing features can be reused for downstream detection with relatively little additional latency.
6. End-to-end latency and deployment cost should be evaluated alongside accuracy when designing constrained perception systems.

---

## Project Outputs

The complete workflow produces:

```text
Synthetic haze generation
        ↓
FOV-aware evaluation
        ↓
DCP baseline
        ↓
Lightweight RGB dehazing
        ↓
RGB + Depth fusion
        ↓
Multi-run reliability analysis
        ↓
Automatic detection annotation
        ↓
Joint Dehaze + Detect model
        ↓
ONNX export
        ↓
Latency + quantization evaluation
```

The goal is a **small, reproducible, deployment-conscious perception kernel** that can serve as a foundation for future embedded surgical-vision research.
