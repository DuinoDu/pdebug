# Real Model Integration Tests

The tests in `test_real_model_integration.py` are opt-in smoke tests for the
manifest-backed model inference nodes. They may clone official repositories,
download model weights, build Docker images, and run CUDA workloads.

Normal unit tests do not run real model inference:

```bash
make test
```

Run all real model integration cases:

```bash
PDEBUG_RUN_MODEL_INTEGRATION=1 make model-integration-test
```

Run a subset while iterating:

```bash
PDEBUG_RUN_MODEL_INTEGRATION=1 \
PDEBUG_MODEL_CASES=moondream,segment_anything \
make model-integration-test
```

Useful environment variables:

- `PDEBUG_MODEL_CACHE`: cache directory for cloned repos, fixtures, and model
  assets. Defaults to `~/.cache/pdebug-model-integration`.
- `PDEBUG_MODEL_CASES`: comma-separated case ids to run.
- `PDEBUG_MODEL_ALLOW_NETWORK=0`: prevent clone/download steps and skip cases
  that need network access.
- `PDEBUG_MODEL_FORCE_CPU=1`: force CPU guards for cases that support CPU.

The case matrix is intentionally explicit. If a model needs a special runtime
such as Docker, CUDA, a compiled extension, or a manual checkpoint layout, the
case should declare that requirement instead of hiding it inside the test body.
