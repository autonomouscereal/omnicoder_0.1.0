# Mobile Sandbox (Android/iOS)

This repository provides a minimal HTTP sandbox for code PAL evaluation that can run:
- Locally inside the main Docker image (compose `sandbox` service)
- Remotely on Android-class devices (ARM64) via a small image
- As a stub on iOS by embedding the same HTTP `/run` API in-app

## Android / ARM64

Build the minimal ARM64 image (requires Docker Buildx):

```bash
docker buildx build --platform linux/arm64 -t omnicoder-sandbox:arm64 -f docker/mobile/arm64-sandbox.Dockerfile .
```

Run on device/edge host (ARM64):

```bash
docker run --rm -p 8088:8088 --name sandbox omnicoder-sandbox:arm64
```

Then, in the main stack, set:

```bash
SANDBOX_REMOTE_URL=http://<device_or_edge_ip>:8088
SFB_CODE_SANDBOX=1
```

## iOS stub

iOS cannot run Docker. Embed a tiny HTTP service in your app that implements:

- POST `/run` with JSON `{code:string, tests:string, timeout:int}`
- Response `{ok:bool, stdout:string, stderr:string}`

Suggestions:
- WASM sandbox (e.g., Pyodide) or restricted Swift runners for testable code paths
- Keep identical schema so `RemoteHTTPSandbox` works unmodified

Point the generator to:

```bash
SANDBOX_REMOTE_URL=http://localhost:<port>
SFB_CODE_SANDBOX=1
```

## Compose service (local dev)

Start a local sandbox alongside the stack:

```bash
docker compose up -d sandbox
```

The generator will use the remote URL when set; otherwise, it falls back to an isolated local runner.

## Automation helpers (Android)

From your host with adb:

```bash
# Host runs sandbox, device apps connect to localhost:8088 via adb reverse
python -m omnicoder.tools.mobile_sandbox --platform android --mode host_reverse --port 8088

# Attempt to start device-hosted sandbox via Termux RunCommandService (requires setup on device)
python -m omnicoder.tools.mobile_sandbox --platform android --mode device --port 8088
```

