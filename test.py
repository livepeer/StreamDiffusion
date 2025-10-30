import os
import gc
import cv2
import torch
import random
from tqdm import tqdm
from torchvision.io import write_jpeg
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
BATCH_SIZE = 8

def pick_idx(diff):
    def generate(diff, low, high):
        upper_bound = min(high, diff)
        if low > upper_bound:
            return upper_bound
        return random.randint(low, upper_bound)

    rand = random.random()
    if rand < 0.70:
        return generate(diff, 1, 3)
    elif rand < 0.85:
        return generate(diff, 4, 5)
    elif rand < 0.95:
        return generate(diff, 6, 10)
    else:
        return generate(diff, 11, diff)


def get_next_frame_ids(length):
    for i in range(length-1):
        yield i, i + pick_idx(length - i - 1)

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
        write_jpeg(frame_tensor, f"{output_dir}/{idx}.jpg")
        frame_tensor = frame_tensor.float() / 255.0
        frames.append(frame_tensor)
        idx += 1
        if idx >= 5000:
            cap.release()
            raise Exception("Reached 5000 frames")
    cap.release()


    frame_tuples = []
    for i, idx in get_next_frame_ids(len(frames)):
        prev_frame = frames[i]
        curr_frame = frames[idx]
        frame_tuples.append((prev_frame, curr_frame, (i, idx)))


    for i in range(0, len(frame_tuples), BATCH_SIZE):
        batch = frame_tuples[i : i+BATCH_SIZE]
        optical_flows = model_inference(batch)
        for j, (_, _, (prev_idx, curr_idx)) in enumerate(batch):
            flow_img = optical_flows[j].cpu()
            if flow_img.dtype != torch.uint8:
                flow_img = (flow_img * 255).to(torch.uint8) if flow_img.max() <= 1.0 else flow_img.to(torch.uint8)
            write_jpeg(flow_img, f"{output_dir}/optical_flow_{prev_idx}_{curr_idx}.jpg")

    

def model_inference(frames_batch):
    with torch.no_grad():
        prev_frames = []
        curr_frames = []
        
        for prev_frame, curr_frame, _ in frames_batch:
            prev_frames.append(prev_frame)
            curr_frames.append(curr_frame)
        
        prev_batch = torch.stack(prev_frames, dim=0).to(device)
        curr_batch = torch.stack(curr_frames, dim=0).to(device)
        
        prev_batch, curr_batch = transforms(prev_batch, curr_batch)
        
        flows = model(prev_batch, curr_batch)[-1]
        flows_img = flow_to_image(flows)
        
        return flows_img


video_path = "/home/user/datasets/celebs/jmD-dnzaxtw_0.mp4"
output_dir = "/home/user/datasets/test/jmD-dnzaxtw_0"

VIDEO_BASE_PATH = "/home/user/datasets/celebs/"
OUTPUT_DIR_BASE = "/home/user/datasets/processed_celebs/"


error_videos = []
for video_path in tqdm(os.listdir(VIDEO_BASE_PATH)):
    try:
        video_id = video_path.replace(".mp4", "")
        output_dir = os.path.join(OUTPUT_DIR_BASE, video_id)
        if os.path.exists(output_dir):
            continue
        os.makedirs(output_dir, exist_ok=True)
        process_video(os.path.join(VIDEO_BASE_PATH, video_path), output_dir)
    except Exception as e:
        error_videos.append(video_path)
        print(f"Error processing video {video_path}: {e}")
    gc.collect()
    torch.cuda.empty_cache()

print(error_videos)