# Week 3 Submission: Style Control with ControlNet
**Intern:** [Your Name]
**Domain:** AI/ML (AI-Powered Fashion Design Assistant)
**Date:** June 15, 2026

## 1. Overview of Work Completed
This week focused on integrating structural constraints into our Stable Diffusion pipeline to ensure generative outputs align with specific fashion sketches and poses. The following tasks were completed:
* **Integrated ControlNet Estimators:** Deployed Canny edge detection, Scribble (sketch-to-design), and OpenPose models alongside the base Stable Diffusion pipeline.
* **Sketch-to-Design Workflow:** Built a functional inference script that takes a raw fashion sketch and a text prompt (e.g., fabric type, color, environment) and outputs a high-fidelity garment preserving the original silhouette.
* **Fine-Tuning Exploration:** Researched fine-tuning ControlNet layers on the DeepFashion dataset to better interpret specific fashion construction lines (seams, pleats, darts).

## 2. Evaluation: Controlled vs. Uncontrolled Generation
To evaluate the effectiveness of the pipeline, we compared standard text-to-image generation against ControlNet-guided generation.

* **Uncontrolled Output (Text Only):** When prompting for "a flowing red silk evening gown," the base Stable Diffusion model generated high-quality images, but the structural design (neckline, drape, sleeve length) was completely randomized and failed to match the designer's intent.
* **Controlled Output (Text + ControlNet):** By feeding a specific sketch through the Canny/Scribble ControlNet annotator alongside the same prompt, the model successfully locked in the garment's boundaries. The output preserved the exact silhouette, pleat lines, and posture of the original sketch while realistically applying the requested "red silk" texture. 

## 3. Technical Challenges & Optimization
* **Conditioning Scale Balancing:** A major challenge was finding the right `controlnet_conditioning_scale`. Setting it to 1.0 sometimes made the final render look too "sketch-like" or artificial. Reducing it to **0.7 - 0.8** yielded the best balance—allowing the Stable Diffusion base model enough creative freedom to render realistic fabrics while still respecting the strict boundaries of the ControlNet guide.
* **Annotator Selection:** For rough fashion illustrations, the `Scribble` annotator outperformed `Canny`, as Canny tends to pick up too much unwanted noise from pencil shading.