import torch
import cv2
import numpy as np
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

DEVICE = "cuda"

def warp_frame(frame, flow):
    h, w = frame.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=DEVICE), 
        torch.arange(w, device=DEVICE), 
        indexing="ij"
    )
    
    flow_map = torch.stack((grid_x, grid_y), dim=2).float()
    flow_map += flow
    
    normalized_flow_map = flow_map.clone()
    normalized_flow_map[..., 0] = 2.0 * normalized_flow_map[..., 0] / max(w - 1, 1) - 1.0
    normalized_flow_map[..., 1] = 2.0 * normalized_flow_map[..., 1] / max(h - 1, 1) - 1.0
    
    warped_frame = torch.nn.functional.grid_sample(
        frame.unsqueeze(0),
        normalized_flow_map.unsqueeze(0),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return warped_frame.squeeze(0)

def calculate_temporal_consistency_raft(video_path):
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(DEVICE)
    model = model.eval()
    transforms = Raft_Large_Weights.DEFAULT.transforms()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1.0

    ret, prev_frame_cv = cap.read()
    if not ret:
        cap.release()
        return -1.0
        
    prev_frame = torch.from_numpy(prev_frame_cv).permute(2, 0, 1).to(DEVICE)
    
    warp_errors = []

    while True:
        ret, current_frame_cv = cap.read()
        if not ret:
            break
            
        current_frame = torch.from_numpy(current_frame_cv).permute(2, 0, 1).to(DEVICE)
        
        prev_frame_transformed, current_frame_transformed = transforms(prev_frame, current_frame)
        
        prev_frame_transformed = prev_frame_transformed.unsqueeze(0)
        current_frame_transformed = current_frame_transformed.unsqueeze(0)
        
        with torch.no_grad():
            list_of_flows = model(prev_frame_transformed, current_frame_transformed)
            predicted_flow = list_of_flows[-1].squeeze(0)
            predicted_flow = predicted_flow.permute(1, 2, 0)

        warped_prev_frame = warp_frame(prev_frame.float(), predicted_flow)
        
        diff = torch.abs(current_frame.float() - warped_prev_frame)
        warp_errors.append(torch.mean(diff).item())
        
        prev_frame = current_frame

    cap.release()
    return np.mean(warp_errors) if warp_errors else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    score = calculate_temporal_consistency_raft(args.video_path)
    
    if score != -1:
        print(f"RAFT Warping Error Score: {score:.4f}")