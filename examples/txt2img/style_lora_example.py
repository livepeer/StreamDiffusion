import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Loading style LoRA example...")
    
    # Hardcoded configuration
    output = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "style_lora_output.png")
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    style_lora_path = r"C:\_dev\comfy\ComfyUI\models\loras\ral-dissolve-sd15.safetensors"
    prompt = "1girl with brown dog hair, thick glasses, smiling, ral-dissolve"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=None,  # No LoRA dict, we'll load manually
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=10,
        acceleration="xformers",
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=2,
        use_lcm_lora=False,  # Disable automatic LCM loading
    )

    pipe = stream.stream.pipe
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
    pipe.load_lora_weights(style_lora_path, adapter_name="ral-dissolve")
    pipe.set_adapters(["lcm", "ral-dissolve"], adapter_weights=[1.0, 1.2])

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    for i in range(stream.batch_size - 1):
        stream()

    output_image = stream()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    output_image.save(output)
    print(f"Image saved to: {output}")


if __name__ == "__main__":
    main() 