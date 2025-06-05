from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision


def denormalize(images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


def pt_to_numpy(images: torch.Tensor):
    if images.dim() == 3:
        images = images.unsqueeze(0)  # Convert [C, H, W] → [1, C, H, W]

    # Debugging: Print the shape before permutation
    print("Debug: Shape before permutation:", images.shape)

    # Ensure the input is in the format [N, C, H, W]
    if images.shape[1] != 3:
        raise ValueError("Expected input shape [N, C, H, W] with C=3, but got shape:", images.shape)

    images = (
        ((images + 1) * 255 / 2)
        .clamp(0, 255)
        .detach()
        .permute(0, 2, 3, 1)  # Ensure correct order to [N, H, W, C]
        .round()
        .type(torch.uint8)
        .cpu()
        .numpy()
    )
    return images

def numpy_to_pil(images: np.ndarray) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    # Ensure the images are in the correct format
    images = images.squeeze()  # Remove any singleton dimensions
    if images.ndim == 2:  # Grayscale image
        images = np.stack([images] * 3, axis=-1)  # Convert to RGB by stacking
    images = (images * 255).round().astype("uint8")

    # Debugging: Print the shape of images before conversion to PIL
    print("Debug: Shape before conversion to PIL:", images.shape)

    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            PIL.Image.fromarray(image.squeeze(), mode="L") for image in images
        ]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    # Return a single image if only one is processed
    if len(pil_images) == 1:
        return pil_images[0]

    return pil_images


def postprocess_image(
    image: torch.Tensor,
    output_type: str = "pil",
    do_denormalize: Optional[List[bool]] = None,
) -> Union[torch.Tensor, np.ndarray, PIL.Image.Image]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )

    if output_type == "latent":
        return image

    do_normalize_flg = True
    if do_denormalize is None:
        do_denormalize = [do_normalize_flg] * image.shape[0]

    image = torch.stack(
        [
            denormalize(image[i]) if do_denormalize[i] else image[i]
            for i in range(image.shape[0])
        ]
    )

    if output_type == "pt":
        return image

    # Convert to NumPy array if not already
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Debugging: Print the shape of the image after conversion to NumPy
    print("Debug: Image shape after pt_to_numpy:", image.shape)

    if output_type == "np":
        return image

    if output_type == "pil":
        pil_images = numpy_to_pil(image)
        # Return a single image if only one is processed
        if len(pil_images) == 1:
            return pil_images[0]
        return pil_images


def process_image(
    image_pil: PIL.Image.Image, range: Tuple[int, int] = (-1, 1)
) -> Tuple[torch.Tensor, PIL.Image.Image]:
    image = torchvision.transforms.ToTensor()(image_pil)
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image[None, ...], image_pil


def pil2tensor(image_pil: PIL.Image.Image) -> torch.Tensor:
    height = image_pil.height
    width = image_pil.width
    imgs = []
    img, _ = process_image(image_pil)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear"
    )
    image_tensors = images.to(torch.float16)
    return image_tensors
