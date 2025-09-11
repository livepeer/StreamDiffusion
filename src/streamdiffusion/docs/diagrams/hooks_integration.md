# Hooks Integration

```mermaid
graph LR
    A[Pipeline Stages] --> B[Embedding Hooks: Prompt Blending]
    B --> C[UNet Hooks: ControlNet/IPAdapter]
    C --> D[Orchestrator Calls: Processors]
    D --> E[Latent/Image Hooks: Pre/Post Processing]
    
    F[StreamParameterUpdater] -.->|Update Configs| C
    G[Config] -->|Register Hooks| B
    G -->|Register Hooks| C
    G -->|Register Hooks| E