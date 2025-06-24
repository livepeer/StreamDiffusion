import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    
    # Hardcoded configuration
    input_image = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "hand_up512.png")
    output = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "style_lora_img2img_output.png")
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    style_lora_path = r"C:\_dev\comfy\ComfyUI\models\loras\ral-dissolve-sd15.safetensors"
    prompt = "1girl with brown dog hair, thick glasses, smiling, ral-dissolve"
    negative_prompt = "low quality, bad quality, blurry, low resolution"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=None,  # No LoRA dict, we'll load manually
        t_index_list=[22, 32, 45],  # img2img t_index
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=10,
        acceleration="xformers",
        mode="img2img",  # Important: img2img mode
        use_denoising_batch=True,  # img2img default
        cfg_type="self",  # img2img can use CFG
        seed=2232341124123,
        use_lcm_lora=False,  # Disable automatic LCM loading
    )

    pipe = stream.stream.pipe
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
    pipe.load_lora_weights(style_lora_path, adapter_name="ral-dissolve")
    pipe.set_adapters(["lcm", "ral-dissolve"], adapter_weights=[1.0, 1.2])

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.2,
        delta=0.5,
    )

    image_tensor = stream.preprocess_image(input_image)

    for i in range(stream.batch_size - 1):
        stream(image=image_tensor)

    output_image = stream(image=image_tensor)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    output_image.save(output)
    print(f"Image saved to: {output}")


if __name__ == "__main__":
    main() 