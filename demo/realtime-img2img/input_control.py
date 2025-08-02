from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import asyncio
import threading
import time
import logging


class InputControl(ABC):
    """Generic interface for input controls that can modify parameters"""
    
    def __init__(self, parameter_name: str, min_value: float = 0.0, max_value: float = 1.0):
        self.parameter_name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        self.is_active = False
        self.update_callback: Optional[Callable[[str, float], None]] = None
    
    @abstractmethod
    async def start(self) -> None:
        """Start the input control"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the input control"""
        pass
    
    @abstractmethod
    def get_current_value(self) -> float:
        """Get the current normalized value (0.0 to 1.0)"""
        pass
    
    def set_update_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for parameter updates"""
        self.update_callback = callback
    
    def normalize_value(self, raw_value: float) -> float:
        """Normalize raw input value to 0.0-1.0 range"""
        return max(0.0, min(1.0, raw_value))
    
    def scale_to_parameter(self, normalized_value: float) -> float:
        """Scale normalized value to parameter range"""
        return self.min_value + (normalized_value * (self.max_value - self.min_value))
    
    def _trigger_update(self, normalized_value: float) -> None:
        """Trigger parameter update if callback is set"""
        if self.update_callback:
            scaled_value = self.scale_to_parameter(normalized_value)
            self.update_callback(self.parameter_name, scaled_value)


# MicrophoneInput moved to frontend - browser handles microphone access


class InputManager:
    """Manages multiple input controls"""
    
    def __init__(self):
        self.inputs: Dict[str, InputControl] = {}
        self.parameter_update_callback: Optional[Callable[[str, float], None]] = None
    
    def add_input(self, input_id: str, input_control: InputControl) -> None:
        """Add an input control"""
        input_control.set_update_callback(self._handle_parameter_update)
        self.inputs[input_id] = input_control
        logging.info(f"InputManager: Added input control {input_id} for parameter {input_control.parameter_name}")
    
    def remove_input(self, input_id: str) -> None:
        """Remove an input control"""
        if input_id in self.inputs:
            asyncio.create_task(self.inputs[input_id].stop())
            del self.inputs[input_id]
            logging.info(f"InputManager: Removed input control {input_id}")
    
    async def start_input(self, input_id: str) -> None:
        """Start a specific input control"""
        if input_id in self.inputs:
            await self.inputs[input_id].start()
    
    async def stop_input(self, input_id: str) -> None:
        """Stop a specific input control"""
        if input_id in self.inputs:
            await self.inputs[input_id].stop()
    
    async def start_all(self) -> None:
        """Start all input controls"""
        for input_control in self.inputs.values():
            await input_control.start()
    
    async def stop_all(self) -> None:
        """Stop all input controls"""
        for input_control in self.inputs.values():
            await input_control.stop()
    
    def set_parameter_update_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for parameter updates from any input"""
        self.parameter_update_callback = callback
    
    def get_input_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all inputs"""
        status = {}
        for input_id, input_control in self.inputs.items():
            status[input_id] = {
                "parameter_name": input_control.parameter_name,
                "is_active": input_control.is_active,
                "current_value": input_control.get_current_value(),
                "min_value": input_control.min_value,
                "max_value": input_control.max_value
            }
        return status
    
    def _handle_parameter_update(self, parameter_name: str, value: float) -> None:
        """Handle parameter update from input controls"""
        if self.parameter_update_callback:
            self.parameter_update_callback(parameter_name, value)