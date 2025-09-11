# Parameter Updating

```mermaid
graph TD
    subgraph "Runtime Update Entry Point"
        A["update_stream_params Call"]
        A --> B["Thread Lock: _update_lock"]
    end
    
    subgraph "Parameter Branches"
        B --> C{"Prompt List Provided?"}
        C -->|Yes| D["_cache_prompt_embeddings: Cache/Encode Prompts"]
        C -->|No| E{"Seed List Provided?"}
        E -->|Yes| F["_cache_seed_noise: Cache/Generate Noise"]
        E -->|No| G{"ControlNet Config Provided?"}
        G -->|Yes| H["Diff Current vs Desired: Add/Remove/Update Scales/Enabled"]
        H --> I["Update ControlNet Pipeline: reorder/add/remove/update_scale"]
        G -->|No| J{"IPAdapter Config Provided?"}
        J -->|Yes| K["Update Scale: Uniform or Per-Layer Vector"]
        K --> L["Set Weight Type: Linear/SLERP for Layers/Steps"]
        J -->|No| M{"Hook Config Provided? e.g., Image/Latent Pre/Post"}
        M -->|Yes| N["Diff Current vs Desired: Modify/Add/Remove Processors In-Place"]
        N --> O["Update Processor Params/Enabled/Order"]
        M -->|No| P["Update Timestep/Resolution: Recalc Scalings/Batches"]
    end
    
    subgraph "Blending & Caching Layer"
        D --> Q["_apply_prompt_blending: Linear/SLERP"]
        F --> R["_apply_seed_blending: Linear/SLERP"]
        I --> S["Cache Stats: Hits/Misses for Monitoring"]
        L --> S
        O --> S
        P --> S
        Q --> T["Update Pipeline Tensors: prompt_embeds/init_noise"]
        R --> T
        S --> T
    end
    
    subgraph "Pipeline Integration"
        T --> U["Pipeline Uses Updated Tensors/Hooks"]
    end
    
    subgraph "Shared Utilities"
        V["Normalize Weights: Sum to 1.0 (Optional)"]
        W["Thread-Safe Lock: Prevent Race Conditions"]
        X["Cache Reindexing: Handle Add/Remove"]
    end
    
    C -.->|"Use"| V
    E -.->|"Use"| V
    B -.->|"Protect"| W
    D -.->|"Use"| X
    F -.->|"Use"| X
    H -.->|"Use"| X
    J -.->|"Use"| X
    M -.->|"Use"| X