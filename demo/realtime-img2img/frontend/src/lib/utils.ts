import * as piexif from "piexifjs";

interface IImageInfo {
    prompt?: string;
    negative_prompt?: string;
    seed?: number;
    guidance_scale?: number;
}

export function snapImage(imageEl: HTMLImageElement, info: IImageInfo) {
    try {
        const zeroth: { [key: string]: any } = {};
        const exif: { [key: string]: any } = {};
        const gps: { [key: string]: any } = {};
        zeroth[piexif.ImageIFD.Make] = "LCM Image-to-Image ControNet";
        zeroth[piexif.ImageIFD.ImageDescription] = `prompt: ${info?.prompt} | negative_prompt: ${info?.negative_prompt} | seed: ${info?.seed} | guidance_scale: ${info?.guidance_scale}`;
        zeroth[piexif.ImageIFD.Software] = "https://github.com/radames/Real-Time-Latent-Consistency-Model";
        exif[piexif.ExifIFD.DateTimeOriginal] = new Date().toISOString();

        const exifObj = { "0th": zeroth, "Exif": exif, "GPS": gps };
        const exifBytes = piexif.dump(exifObj);

        const canvas = document.createElement("canvas");
        canvas.width = imageEl.naturalWidth;
        canvas.height = imageEl.naturalHeight;
        const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
        ctx.drawImage(imageEl, 0, 0);
        const dataURL = canvas.toDataURL("image/jpeg");
        const withExif = piexif.insert(exifBytes, dataURL);

        const a = document.createElement("a");
        a.href = withExif;
        a.download = `lcm_txt_2_img${Date.now()}.png`;
        a.click();
    } catch (err) {
        console.log(err);
    }
}

export interface ResolutionInfo {
  width: number;
  height: number;
  aspectRatio: number;
  aspectRatioString: string;
}

export function parseResolution(resolutionString: string): ResolutionInfo {
  const match = resolutionString.match(/(\d+)x(\d+)/);
  if (!match) {
    // Default fallback
    return {
      width: 512,
      height: 512,
      aspectRatio: 1,
      aspectRatioString: "1:1"
    };
  }
  
  const width = parseInt(match[1], 10);
  const height = parseInt(match[2], 10);
  const aspectRatio = width / height;
  
  // Calculate simplified aspect ratio string
  const gcd = (a: number, b: number): number => b === 0 ? a : gcd(b, a % b);
  const divisor = gcd(width, height);
  const ratioW = width / divisor;
  const ratioH = height / divisor;
  const aspectRatioString = `${ratioW}:${ratioH}`;
  
  return {
    width,
    height,
    aspectRatio,
    aspectRatioString
  };
}

export function calculateFitDimensions(
  containerWidth: number,
  containerHeight: number,
  targetWidth: number,
  targetHeight: number
): { width: number; height: number; scale: number } {
  const containerRatio = containerWidth / containerHeight;
  const targetRatio = targetWidth / targetHeight;
  
  let width, height, scale;
  
  if (containerRatio > targetRatio) {
    // Container is wider than target - fit to height
    height = containerHeight;
    width = height * targetRatio;
    scale = height / targetHeight;
  } else {
    // Container is taller than target - fit to width
    width = containerWidth;
    height = width / targetRatio;
    scale = width / targetWidth;
  }
  
  return { width, height, scale };
}

export function calculateCropRegion(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
): { x: number; y: number; width: number; height: number } {
  const sourceRatio = sourceWidth / sourceHeight;
  const targetRatio = targetWidth / targetHeight;
  
  if (sourceRatio > targetRatio) {
    // Source is wider - crop horizontally
    const cropWidth = sourceHeight * targetRatio;
    return {
      x: (sourceWidth - cropWidth) / 2,
      y: 0,
      width: cropWidth,
      height: sourceHeight
    };
  } else {
    // Source is taller - crop vertically
    const cropHeight = sourceWidth / targetRatio;
    return {
      x: 0,
      y: (sourceHeight - cropHeight) / 2,
      width: sourceWidth,
      height: cropHeight
    };
  }
}
