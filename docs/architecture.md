# Architecture

```mermaid
flowchart TD
    A[Official labeled train images] --> B[Decode audit and pixel hash deduplication]
    T[Official test images - held out] --> B
    B --> C[Stratified train / validation manifest]
    C --> D[Train-only mirror and small rotation]
    D --> E[Frozen ImageNet ResNet-18 - 512 features]
    E --> F[Training-only standardization and balanced logistic head]
    F --> G[Validation C and threshold selection]
    G --> H[Locked model and metadata / SHA-256]
    H --> I[One final test evaluation and error analysis]
    H --> J[FastAPI startup - integrity check and warmup]
    R[Browser inspection workspace - upload or reference image] --> K
    K[JPEG or PNG multipart upload] --> L[Body limit / MIME / decode / dimension checks]
    L --> M[Shared resize and normalization]
    J --> N[Bounded CPU inference]
    M --> N
    N --> O[Defect score and validation-selected threshold]
    O --> P[Class / confidence / version / request ID]
    N --> Q[Request timing and error logs]
```

Training and inference share `casting_inspector/model.py`. The deployed model
contains the feature extractor, normalization buffers and trained head. Serving
does not require scikit-learn, network downloads or access to the dataset.

One process with two Torch threads and one active model call bounds CPU
contention. Uvicorn bounds concurrent connections; run additional replicas for
throughput after load testing. Readiness follows successful artifact loading;
liveness confirms that the process responds. A reverse proxy should terminate
TLS, authenticate clients, rate-limit requests and enforce an upload timeout.
