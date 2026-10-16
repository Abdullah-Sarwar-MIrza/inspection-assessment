# Docker deployment and operation

The container includes the inference API, browser interface and trained model.
A native Python installation and the training dataset are not required for
Docker inference. Docker Desktop uses the Linux-container engine on Windows.

## Start the application

Start Docker Desktop and wait until its engine is running. Open PowerShell in
the repository folder. In the current Windows workspace:

```powershell
cd "D:\Job Assessment"
$env:INSPECTOR_PORT = "8001"
docker compose up --build -d --wait --wait-timeout 90
```

The first build downloads the Python base image and runtime dependencies.
Subsequent starts reuse the built image. `-d` runs the service in the background;
closing PowerShell does not stop it. `--wait` waits for the health check.

Open **http://127.0.0.1:8001/** for the inspection interface and
**http://127.0.0.1:8001/docs** for interactive API documentation.

These instructions use port 8001 on Windows. The app listens on port 8000 inside
the container. `INSPECTOR_PORT` controls the host port; without a setting it
defaults to 8000. A local `.env` file containing `INSPECTOR_PORT=8001` persists
that choice across terminal sessions. Local `.env` settings are not committed.

## Confirm the container is healthy

```powershell
docker compose ps
curl.exe --fail --silent --show-error http://127.0.0.1:8001/health/ready
```

`docker compose ps` should show the inspector service as healthy and a mapping
from `127.0.0.1:8001` to container port `8000`. Readiness returns:

```json
{"status":"ready","model_version":"resnet18-frozen-v1"}
```

## Inspect an image in the browser

1. Open http://127.0.0.1:8001/.
2. Select **browse files** or drag a JPG/PNG onto the upload panel.
3. Select **Inspect image**.
4. Review the class, confidence, defect score and latency.
5. **Save result** downloads the JSON response.

Reference images in the interface run a real prediction when selected. Local
examples are available in `reports/examples/`: `normal_1.jpg`, `normal_2.jpg`,
`defective_1.jpg` and `defective_2.jpg`.

Confidence is an uncalibrated model score, not a guarantee of correctness.

## Inspect an image through the API

Run from the repository folder:

```powershell
curl.exe --fail --silent --show-error -X POST http://127.0.0.1:8001/predict -F "file=@reports/examples/defective_1.jpg;type=image/jpeg"
```

The verified Docker smoke test returned Defective with a defect score of
approximately 0.64556 and request latency of 35.349 ms. This is one successful
request, not a container latency benchmark. Request IDs and timings vary.
Recorded evidence is in `reports/docker_verification.json`.

The response contains `predicted_class`, `confidence`, `defect_probability`,
`threshold`, `model_version`, `request_id` and `latency_ms`.

## View logs

```powershell
docker compose logs --tail 50 inspector
docker compose logs -f inspector
```

Press Ctrl+C to stop following logs. This does not stop the container.
Logs include request identifiers, response status and elapsed time.

## Stop, resume or rebuild

Stop the application while preserving its container and image:

```powershell
docker compose stop inspector
```

Resume the already-built application:

```powershell
docker compose up -d --wait --wait-timeout 90
```

Rebuild after application, model or dependency changes:

```powershell
docker compose up --build -d --wait --wait-timeout 90
```

Remove the Compose container and network:

```powershell
docker compose down
```

The repository files and trained model remain on the host.

## Configuration and operating limits

| Setting | Value |
|---|---|
| Browser/API host port in this guide | 8001 |
| Container port | 8000 |
| Model files | Bundled under `/app/artifacts` |
| Operating system | Linux container |
| Runtime account | Non-root `inspector` |
| Root filesystem | Read-only |
| Temporary storage | 64 MiB tmpfs at `/tmp` |
| Memory limit | 1 GiB |
| CPU limit | 2 CPUs |
| Torch threads | 2 |
| Active inference calls | 1 per process |
| Uvicorn concurrency limit | 16 |
| Upload formats | JPEG and PNG |
| Upload file limit | 8 MiB |

The port is bound to loopback, so the service is available on this computer.
External deployment requires appropriate network access controls, TLS,
authentication and load testing. This assessment does not establish a production
SLA or factory acceptance criteria.

## Troubleshooting

| Symptom | Action |
|---|---|
| `docker` is not recognized | Reopen PowerShell after installation; confirm Docker Desktop is installed. |
| Cannot connect to the Docker engine | Start Docker Desktop and wait for its engine to become ready. |
| Port is already allocated | Set `$env:INSPECTOR_PORT = "8002"`, rerun Compose, and use port 8002 in browser/API URLs. |
| Container is unhealthy or exits | Inspect `docker compose logs --tail 100 inspector`. |
| File upload returns 415 | Submit a JPEG or PNG with the matching media type. |
| File upload returns 413 | Use an image below 8 MiB. |
| File upload returns 422 | Use a valid image, at least 32 pixels per side and no larger than 16 megapixels. |
| Code changes are not visible | Rebuild with `docker compose up --build -d --wait`, then refresh the browser. |

## Verification scope

The image built successfully, the service became healthy, readiness returned
HTTP 200, and a real JPEG upload returned a successful Defective prediction.
The existing 16 inference/API checks and 12 browser checks were performed in
the native environment; they are separate from this Docker smoke test.
