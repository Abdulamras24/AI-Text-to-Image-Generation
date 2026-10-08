# Task 1 – Pretrained Text-to-Image Model Refinement Using LoRA

## Objective

The objective of this task was to refine a pretrained text-to-image model using a custom image-caption dataset. Stable Diffusion 1.5 was used as the pretrained model, with LoRA for parameter-efficient fine-tuning.

## Dataset

The custom dataset used was the Digimon BLIP Captions dataset from Hugging Face.

- Image-text pairs: 1,071
- Image format: RGB
- Training resolution: 512 x 512
- Captions: Text descriptions corresponding to the images

## Preprocessing

The images were converted to RGB format and resized to 512 x 512 pixels.

A metadata.jsonl file was created to associate each image with its corresponding caption.

The final dataset was verified using the Hugging Face imagefolder loader with the columns image and text.

## Pretrained Model

Model: runwayml/stable-diffusion-v1-5

The pretrained model was loaded using the Diffusers library and tested before training.

## LoRA Training

LoRA was used for parameter-efficient fine-tuning.

Training configuration:

- Base model: Stable Diffusion 1.5
- Dataset size: 1,071 images
- Resolution: 512 x 512
- Batch size: 1
- Gradient accumulation: 4
- Epochs: 1
- Learning rate: 0.0001
- LoRA rank: 4
- Precision: FP16
- GPU: NVIDIA Tesla T4
- Optimization steps: 268

## Training Result

Training completed successfully for one epoch.

Final reported step loss: 0.165

The trained LoRA weights were saved as:

task1_lora_output/pytorch_lora_weights.safetensors

## Image Generation

The trained LoRA adapter was successfully loaded into Stable Diffusion 1.5 and used to generate an image.

Example prompt:

"a drawing of a cartoon character holding a wrench"

## Base Model vs LoRA

The same prompt was used with the base model and the trained LoRA model.

The following files were created:

- task1_base_image.png
- task1_lora_image.png
- task1_base_vs_lora_comparison.png

The comparison provides a visual demonstration of the outputs before and after applying the trained LoRA adapter.

## Limitations

This was a proof-of-concept experiment using one training epoch and a relatively small custom dataset. Therefore, the results should not be interpreted as a comprehensive measurement of model improvement.

More training epochs, additional prompts, and quantitative or human evaluation could be used for a more detailed assessment.

## Conclusion

The task successfully demonstrated the process of refining a pretrained text-to-image model using a custom image-caption dataset and LoRA.

The trained LoRA adapter was successfully saved, loaded, and used to generate an image with Stable Diffusion 1.5.
