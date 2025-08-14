from __future__ import annotations

from typing import Optional

from .preprocessing_orchestrator import PreprocessingOrchestrator


class OrchestratorUser:
    """
    Minimal base class to attach a shared PreprocessingOrchestrator from the stream.
    No convenience methods; strictly enforces presence of a shared orchestrator on stream.
    """

    _preprocessing_orchestrator: Optional[PreprocessingOrchestrator] = None

    def attach_orchestrator(self, stream) -> None:
        orchestrator = getattr(stream, 'preprocessing_orchestrator', None)
        if orchestrator is None:
            # Lazy-create on stream once, on first user that needs it
            orchestrator = PreprocessingOrchestrator(device=stream.device, dtype=stream.dtype, max_workers=4)
            setattr(stream, 'preprocessing_orchestrator', orchestrator)
        self._preprocessing_orchestrator = orchestrator


