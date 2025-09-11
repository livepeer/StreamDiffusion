# Orchestrator Flow

```mermaid
graph TB
    subgraph "Input Layer - Distinct Preprocessing Types"
        A["ControlNet/IPAdapter Inputs: Raw Images for Module Preprocessing"]
        B["Pipeline Hooks: Latent/Image Tensors for Hook Stages"]
        C["Postprocessing: VAE Output Images for Enhancement"]
    end
    
    subgraph "PreprocessingOrchestrator (ControlNet/IPAdapter - Intraframe Parallelism)"
        D["Raw Images: Multiple ControlNets/IPAdapters"]
        E["Group by Processor Type: e.g., All Canny Processors Grouped"]
        F["Intraframe Parallel: ThreadPoolExecutor per Group"]
        F --> G["Process Group in Parallel: e.g., Canny for CN1 and CN2 Simultaneously"]
        G --> H["Merge/Broadcast Group Results to Specific Modules e.g. Canny to CN1 and CN2"]
        I["Intraframe Sequential: Unique Processors Single Thread"]
        H --> J["Cache by Type: Reuse Across Modules/Frames"]
        I --> J
        J --> K["Output Distinct Tensors for Each ControlNet/IPAdapter"]
    end
    
    subgraph "PipelinePreprocessingOrchestrator (Hook Stages - Sequential Chain)"
        L["Latent/Image Tensors from Pipeline Hooks"]
        M["Sequential Chain: _execute_pipeline_chain"]
        M --> N["Single Processor Application: e.g., Latent Feedback Sequential"]
        N --> O["Next Processor in Order (order attr)"]
        O --> P["Chain Continues: No Parallelism Within Chain"]
        P --> M
        Q["Output Processed Tensor to Next Pipeline Hook/Stage"]
    end
    
    subgraph "PostprocessingOrchestrator (Output - Cached Sequential)"
        R["VAE Decoded Images"]
        S["Sequential with Cache Check: _apply_single_postprocessor"]
        S --> T{"Cache Hit for Identical Input?"}
        T -->|Yes| U["Reuse Cached: e.g., Same Upscale Params"]
        T -->|No| V["Process Sequential: Realesrgan_trt then Sharpen"]
        U --> W["Output Enhanced Image"]
        V --> W
    end
    
    subgraph "BaseOrchestrator (All Types - Interframe Pipelining)"
        X{"Use Sync Processing? (Feedback/Temporal Config)"}
        X -->|Yes| Y["Process Sync: Sequential/Immediate (No Lag, Low Throughput)"]
        X -->|No| Z["Background Thread: Pipelined/1-Frame Lag (High Throughput)"]
        Y --> AA["Apply Current Frame Results"]
        Z --> AA
        AA --> BB["Output to Pipeline/Next Orchestrator/Stage"]
    end
    
    subgraph "Shared Resources & Integration"
        CC["OrchestratorUser Mixin: Attach Shared Orchestrators to Modules/Hooks"]
        DD["StreamParameterUpdater: Runtime Param Updates to Processors"]
        EE["Thread Lock: Ensure Thread-Safe Parallel & Pipelined Execution"]
    end
    
    A --> E
    B --> M
    C --> S
    E --> X
    M --> X
    S --> X
    CC -.->|"Shared Orchestrators"| E
    CC -.->|"Shared Orchestrators"| M
    CC -.->|"Shared Orchestrators"| S
    DD -.->|"Dynamic Params"| E
    DD -.->|"Dynamic Params"| M
    DD -.->|"Dynamic Params"| S
    EE -.->|"Protect"| F
    EE -.->|"Protect"| M
    EE -.->|"Protect"| S