## Summary

I identified and fixed a batch-slicing defect in the Ollama embedding provider that caused overlapping batches when `max_batch_size` was configured to a value other than the default of 32.

I also introduced configurable batch-size support, added dedicated unit tests, and created a benchmark script to validate embedding integrity across different batch sizes.

## Problem

In `ai/tutor_ai/providers/ollama.py`, the `embed()` method used `self.max_batch_size` as the iteration step but relied on the hardcoded `MAX_EMBED_BATCH` constant (32) for the slice boundary.

When `max_batch_size=16`, batches overlapped:

- First iteration: `texts[0:32]`
- Second iteration: `texts[16:48]`

This caused duplicate embedding computation and could result in input/output vector-count mismatches.

## Changes Made

### 1. Fixed embedding batch slicing
- Replaced the hardcoded `MAX_EMBED_BATCH` slice boundary with `self.max_batch_size`.
- Ensured consistent batch iteration and slicing.

### 2. Added configurable batch size
- Added `ollama_max_batch_size` to `Settings`.
- Introduced `TUTOR_OLLAMA_MAX_BATCH_SIZE`.
- Added validation restricting values to 1–64.
- Updated `build_provider()` to pass the configured batch size to the Ollama provider.

### 3. Updated environment documentation
- Documented `TUTOR_OLLAMA_MAX_BATCH_SIZE=32` in `.env.example`.
- Added guidance for resource-constrained environments.

### 4. Added unit tests
Created `ai/tests/unit/test_batch_slicing_optimization.py` with nine tests covering:
- Batch size 16 without overlapping slices.
- Batch sizes 1, 7, 10, 32, and 64.
- Partial trailing batches.
- Batch-size clamping.
- Settings validation.
- Provider factory configuration.

### 5. Added local Ollama benchmark
Created `benchmark_batch_slicing.py` to evaluate `nomic-embed-text` across batch sizes 8, 16, and 32.

## Verification and Test Results

**Dedicated unit tests:**  
`9 passed in 0.87s`

**Full unit test suite:**  
`163 passed in 2.98s`

**Local Ollama benchmark:**

| Batch Size | Batches | Output Vectors | Execution Time |
|---:|---:|---:|---:|
| 8 | 4 | 32 | 10,331.99 ms |
| 16 | 2 | 32 | 8,238.02 ms |
| 32 | 1 | 32 | 6,369.46 ms |

All tested configurations returned exactly 32 vectors for 32 input passages without duplicate vectors or slice overlap.

## Impact

- Prevents overlapping embedding batches.
- Avoids duplicate embedding computation caused by the slicing defect.
- Supports configurable batch sizes for different hardware environments.
- Improves embedding reliability and adds regression test coverage.

## Testing Checklist

- [x] Fixed hardcoded embedding slice boundary.
- [x] Added configurable Ollama batch size.
- [x] Updated `.env.example`.
- [x] Added nine dedicated unit tests.
- [x] Full unit test suite passed (163 tests).
- [x] Benchmarked batch sizes 8, 16, and 32.
- [x] Verified input/output vector count consistency.

## Conclusion

This PR fixes the Ollama embedding batch overlap issue, introduces configurable batch sizing, and adds regression tests to prevent the defect from recurring.

All dedicated tests and the full unit test suite passed, and local benchmarking confirmed correct output vector counts across the tested configurations.
