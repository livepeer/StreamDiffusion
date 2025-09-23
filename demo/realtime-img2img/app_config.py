"""
Application configuration and settings for realtime-img2img
"""
import yaml
import logging
from pathlib import Path

def load_controlnet_registry():
    """Load ControlNet registry from config file"""
    try:
        registry_path = Path(__file__).parent / "controlnet_registry.yaml"
        with open(registry_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract the available_controlnets section
        return config_data.get('available_controlnets', {})
    except Exception as e:
        logging.exception(f"load_controlnet_registry: Failed to load ControlNet registry: {e}")
        # Fallback to empty registry
        return {}

def load_default_settings():
    """Load default settings from YAML config file"""
    try:
        registry_path = Path(__file__).parent / "controlnet_registry.yaml"
        with open(registry_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data.get('defaults', {})
    except Exception as e:
        logging.exception(f"load_default_settings: Failed to load default settings: {e}")
        # Fallback to hardcoded defaults
        return {
            'guidance_scale': 1.1,
            'delta': 0.7,
            'num_inference_steps': 50,
            'seed': 2,
            't_index_list': [35, 45],
            'ipadapter_scale': 1.0,
            'normalize_prompt_weights': True,
            'normalize_seed_weights': True,
            'prompt': "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        }

# Load configuration at module level
AVAILABLE_CONTROLNETS = load_controlnet_registry()
DEFAULT_SETTINGS = load_default_settings()
