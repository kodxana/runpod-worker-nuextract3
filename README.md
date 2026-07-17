# NuExtract3 Runpod Worker

[![Runpod](https://api.runpod.io/badge/kodxana/runpod-worker-nuextract3)](https://console.runpod.io/hub/kodxana/runpod-worker-nuextract3)

A queue-based Runpod Serverless worker for
[`numind/NuExtract3`](https://huggingface.co/numind/NuExtract3). It performs
structured extraction, document-to-Markdown conversion, and extraction-template
generation directly with Transformers. There is no application HTTP server and
no vLLM sidecar.

## Features

- Text, HTTPS URL, strict base64, PDF, PNG, JPEG, and WebP inputs
- Structured JSON extraction with the native NuExtract template grammar
- Markdown conversion and template generation
- Optional NuExtract reasoning with deterministic sampling settings

### Per-job generation controls

These settings are supplied under `input.generation` and apply only to that job:

| Field | Type | Default | Bounds and behavior |
| --- | --- | --- | --- |
| `thinking` | boolean | `false` | Enables NuExtract reasoning |
| `return_reasoning` | boolean | `false` | Requires `thinking=true` |
| `max_new_tokens` | integer | `1024` | 1 to 4,096 |
| `temperature` | number | `0` non-thinking, `0.6` thinking | 0 to 2; zero is greedy, positive values enable sampling |
| `top_p` | number | `1.0` | Greater than 0 and at most 1; used when sampling |
| `top_k` | integer | `0` | 0 to 100; zero disables top-k filtering |
| `seed` | integer | `0` | 0 to 2,147,483,647; used when sampling |

The model is fixed to the baked NuExtract3 revision, so there is no per-job model
selector. Beam search and unrestricted generation arguments are intentionally not
exposed because they multiply compute or memory and are not recommended by the
model author.

### Structured extraction

```json
{
  "input": {
    "schema_version": "1",
    "mode": "structured",
    "sources": [
      {
        "type": "text",
        "text": "Yesterday I bought apples at Trader Joe's for $12.40."
      }
    ],
    "template": {
      "store": "verbatim-string",
      "total": "number",
      "currency": ["USD", "EUR", "GBP"],
      "items": [
        {
          "name": "verbatim-string",
          "quantity": "integer"
        }
      ]
    },
    "instructions": "Use null when a scalar value is absent.",
    "generation": {
      "thinking": false,
      "return_reasoning": false,
      "max_new_tokens": 1024,
      "temperature": 0.2,
      "top_p": 1.0,
      "top_k": 0,
      "seed": 0
    },
    "output": {
      "delivery": "inline",
      "presign_ttl_seconds": 0
    }
  }
}
```

NuExtract template constructors are:

- Scalar types such as `verbatim-string`, `string`, `integer`, `number`,
  `date`, `currency`, and the other types documented in
  [`TYPES.md`](https://huggingface.co/numind/NuExtract3/blob/2e9fca82ee641e6bb6e1f5d905241e994be27a07/TYPES.md)
- Arrays such as `["string"]`
- Enums such as `["yes", "no", "unknown"]`
- Multi-enums such as `[["A", "B", "C"]]`

### Markdown

```json
{
  "input": {
    "schema_version": "1",
    "mode": "markdown",
    "sources": [
      {
        "type": "url",
        "url": "https://documents.example/invoice.pdf",
        "media_type": "application/pdf",
        "pages": [
          {"start": 1, "end": 2},
          {"start": 5, "end": 5}
        ]
      }
    ],
    "generation": {
      "thinking": true,
      "return_reasoning": true,
      "max_new_tokens": 4096,
      "temperature": 0.6,
      "seed": 0
    }
  }
}
```

PDF page numbers are one-based. Ranges are inclusive and must be sorted,
non-overlapping, and collectively select no more than six pages across the job.
PDFs with more than six pages require explicit ranges.

### Base64 image

```json
{
  "input": {
    "schema_version": "1",
    "mode": "markdown",
    "sources": [
      {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgoAAA..."
      }
    ]
  }
}
```

The `data` value is raw RFC 4648 base64. Data URLs, whitespace, missing padding,
and non-alphabet characters are not accepted.

### Template generation

```json
{
  "input": {
    "schema_version": "1",
    "mode": "template-generation",
    "sources": [
      {
        "type": "text",
        "text": "Create a template for the key details in a rental contract."
      }
    ]
  }
}
```

`template` and `instructions` are only accepted in `structured` mode. Thinking
is not accepted in `template-generation` mode. `return_reasoning` requires
`thinking` to be true.

## Success Response

Inline structured and template-generation results use `result.data`; Markdown
uses `result.text`.

```json
{
  "schema_version": "1",
  "request_id": "job-id",
  "model": {
    "id": "numind/NuExtract3",
    "revision": "2e9fca82ee641e6bb6e1f5d905241e994be27a07"
  },
  "mode": "structured",
  "result": {
    "delivery": "inline",
    "content_type": "application/json",
    "bytes": 55,
    "sha256": "...",
    "data": {
      "store": "Trader Joe's",
      "total": 12.4
    }
  },
  "usage": {
    "input_tokens": 120,
    "generated_tokens": 24,
    "images": 0,
    "pdf_pages": 0,
    "source_bytes": 0,
    "rendered_pixels": 0,
    "finish_reason": "eos",
    "preprocess_ms": 18,
    "generation_queue_ms": 0,
    "generation_ms": 628,
    "postprocess_ms": 4,
    "generated_tokens_per_second": 38.217,
    "gpu_peak_allocated_bytes": 9450000000,
    "gpu_peak_reserved_bytes": 9700000000,
    "inference_ms": 650,
    "duration_ms": 654
  }
}
```

When requested, `reasoning` is a top-level string. It is never returned unless
both `thinking` and `return_reasoning` are true.

The timing fields separate processor work, time waiting for the single CUDA
owner, generation, and output decoding/validation. GPU peak values include the
resident model weights. `generated_tokens_per_second` uses generation time and
includes the terminal EOS token.

## Error Response

Runpod treats a returned `error` field specially, so the structured error is
serialized as its JSON string value:

```json
{
  "error": "{\"schema_version\":\"1\",\"request_id\":\"job-id\",\"code\":\"INVALID_REQUEST\",\"message\":\"Request does not match the schema at /mode.\",\"retryable\":false}"
}
```

Callers can JSON-decode the `error` string. Retryable infrastructure failures
are marked with `retryable: true`. A GPU out-of-memory error also asks Runpod to
refresh the worker.

## S3 Delivery

Set `output.delivery` to `s3`, or use `auto` to upload only when the serialized
result exceeds the inline limit. Runpod may inject credentials at the top level:

```json
{
  "input": {
    "schema_version": "1",
    "mode": "markdown",
    "sources": [{"type": "text", "text": "Document text"}],
    "output": {"delivery": "s3", "presign_ttl_seconds": 600}
  },
  "s3Config": {
    "accessId": "ACCESS_KEY",
    "accessSecret": "SECRET_KEY",
    "bucketName": "results-bucket",
    "endpointUrl": "https://s3.example.com"
  }
}
```

Alternatively configure `OUTPUT_S3_BUCKET` and the standard AWS credential
variables. Object keys are deterministic for the job date, sanitized job ID,
content hash, and output type. The response includes bucket, key, ETag, byte
count, content type, and SHA-256. A presigned URL is included only when
`presign_ttl_seconds` is nonzero.

Every custom endpoint, including an endpoint injected through `s3Config`, must
match an exact host in `S3_ENDPOINT_HOST_ALLOWLIST`. Wildcards are rejected so
the deployment operator, rather than a job payload, defines the S3 trust boundary.
Custom endpoints must be root HTTPS URLs and use path-style bucket addressing.
The worker signs these requests with AWS SigV4 and connects directly to the
validated address while preserving TLS SNI and the HTTP `Host` header. Standard
AWS S3 delivery ignores SDK endpoint overrides from environment and shared config.

## Fixed Limits

| Resource | Limit |
| --- | ---: |
| Sources per request | 8 |
| Combined text | 200,000 characters |
| Combined binary sources | 16 MiB |
| Decoded base64 source | 7,000,000 bytes |
| PDF pages per job | 6 |
| PDF render resolution | 170 DPI |
| Raw image pixels | 25,000,000 |
| Image width or height | 8,192 pixels |
| Processed pixels per image | 4,194,304 |
| Combined processed pixels | 24,000,000 |
| Template | 65,536 bytes, 1,000 nodes, depth 12 |
| Model context | 32,768 tokens |
| Generated output | 4,096 tokens |
| Parsed model JSON | 1,000,000 bytes |
| Inline result | 1,000,000 bytes |

These limits are constants and cannot be weakened with environment variables.

## Environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `SOURCE_HOST_ALLOWLIST` | No | Comma-separated exact hosts or `*.example.com` patterns |
| `S3_ENDPOINT_HOST_ALLOWLIST` | For custom endpoints | Comma-separated exact custom S3 hosts; wildcards are rejected |
| `OUTPUT_S3_BUCKET` | No | Environment-backed result bucket |
| `OUTPUT_S3_ENDPOINT_URL` | No | HTTPS S3-compatible endpoint |
| `OUTPUT_S3_REGION` | No | S3 region; defaults to `AWS_REGION` or `us-east-1` |
| `AWS_ACCESS_KEY_ID` | No | S3 access key; use with `AWS_SECRET_ACCESS_KEY` |
| `AWS_SECRET_ACCESS_KEY` | No | S3 secret key |
| `AWS_SESSION_TOKEN` | No | Optional temporary-credential token |
| `GENERATION_TIMEOUT_SECONDS` | No | Generation limit from 30 to 900; default 600 |
| `DOWNLOAD_CONNECT_TIMEOUT_SECONDS` | No | Connect timeout from 1 to 30; default 5 |
| `DOWNLOAD_READ_TIMEOUT_SECONDS` | No | Read timeout from 1 to 120; default 30 |
| `NUEXTRACT_DEFAULT_THINKING` | No | Endpoint default for reasoning; `true` or `false`, default `false` |
| `NUEXTRACT_DEFAULT_RETURN_REASONING` | No | Endpoint default for returning reasoning; requires default thinking, default `false` |
| `NUEXTRACT_DEFAULT_MAX_NEW_TOKENS` | No | Endpoint output-token default from 1 to 4,096; default 1,024 |
| `NUEXTRACT_DEFAULT_TEMPERATURE` | No | Endpoint sampling default from 0 to 2; when unset, uses 0 non-thinking or 0.6 thinking |
| `NUEXTRACT_DEFAULT_TOP_P` | No | Endpoint top-p default greater than 0 and at most 1; default 1 |
| `NUEXTRACT_DEFAULT_TOP_K` | No | Endpoint top-k default from 0 to 100; default 0 |
| `NUEXTRACT_DEFAULT_SEED` | No | Endpoint sampling seed from 0 to 2,147,483,647; default 0 |


## Build and Run

The image is pinned to PyTorch 2.10, CUDA 12.8, cuDNN 9, Transformers 5.5.4,
Flash Linear Attention 0.5.1, and hash-pinned `causal-conv1d` and pypdfium2
wheels. The same file locks the complete resolved Python 3.12 dependency graph,
including transitive packages.

```bash
docker build --tag runpod-worker-nuextract3:local .
```

Run the SDK's local API mode on a compatible NVIDIA host:

```bash
docker run --rm --gpus all \
  --publish 8000:8000 \
  runpod-worker-nuextract3:local \
  python -u handler.py --rp_serve_api --rp_api_host 0.0.0.0
```

## License

Worker source is licensed under Apache-2.0. NuExtract3 and all dependencies keep
their own license terms; see `NOTICE` and `THIRD_PARTY_NOTICES.md`.
