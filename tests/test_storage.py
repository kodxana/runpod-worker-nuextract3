from __future__ import annotations

import hashlib

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from nuextract_worker.config import MODEL_REVISION, Settings
from nuextract_worker.errors import WorkerError
from nuextract_worker.storage import S3Config, S3Uploader, resolve_s3_config


def _public_resolver(*args, **kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def _private_resolver(*args, **kwargs):
    return [(2, 1, 6, "", ("127.0.0.1", 443))]


def test_per_job_s3_config_is_strict(monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_S3_REGION", "eu-west-1")
    config = resolve_s3_config(
        {
            "s3Config": {
                "accessId": "access",
                "accessSecret": "secret",
                "bucketName": "bucket-name",
                "endpointUrl": "https://s3.example.com",
            }
        }
    )
    assert config == S3Config(
        bucket="bucket-name",
        endpoint_url="https://s3.example.com",
        region="eu-west-1",
        access_key="access",
        secret_key="secret",
    )

    monkeypatch.delenv("OUTPUT_S3_REGION")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    assert (
        resolve_s3_config(
            {
                "s3Config": {
                    "accessId": "access",
                    "accessSecret": "secret",
                    "bucketName": "bucket-name",
                    "endpointUrl": "https://s3.example.com",
                }
            }
        ).region
        == "ap-southeast-2"
    )

    with pytest.raises(WorkerError) as caught:
        resolve_s3_config({"s3Config": {"bucketName": "bucket-name"}})
    assert caught.value.code == "INVALID_S3_CONFIG"


def test_environment_s3_config_requires_paired_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_S3_BUCKET", "bucket-name")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(WorkerError, match="Both AWS access"):
        resolve_s3_config({})

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    config = resolve_s3_config({})
    assert config.access_key == "access"
    assert config.secret_key == "secret"


class _S3Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.put = None

    def put_object(self, **kwargs):
        if self.fail:
            raise RuntimeError("credentials must not escape")
        self.put = kwargs
        return {"ETag": '"etag-value"'}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presign = (operation, Params, ExpiresIn)
        return "https://s3.example.com/presigned"


def test_s3_upload_validates_endpoint_and_returns_integrity_metadata() -> None:
    client = _S3Client()
    uploader = S3Uploader(
        Settings(s3_endpoint_host_allowlist=("s3.example.com",)),
        client_factory=lambda config: client,
        resolver=_public_resolver,
    )
    body = b"# Result"
    digest = hashlib.sha256(body).hexdigest()
    result = uploader.upload(
        config=S3Config("bucket-name", "https://s3.example.com", "us-east-1"),
        job_id="unsafe/job id",
        mode="markdown",
        body=body,
        content_type="text/markdown; charset=utf-8",
        sha256=digest,
        presign_ttl_seconds=600,
    )

    assert result["delivery"] == "s3"
    assert result["etag"] == "etag-value"
    assert result["url"] == "https://s3.example.com/presigned"
    assert "/unsafe_job_id/" in result["key"]
    assert result["key"].endswith(f"/{digest[:16]}.md")
    assert client.put["Metadata"]["model-revision"] == MODEL_REVISION
    assert client.put["Metadata"]["sha256"] == digest


def test_s3_upload_rejects_private_endpoint_and_invalid_bucket() -> None:
    uploader = S3Uploader(
        Settings(s3_endpoint_host_allowlist=("private.example",)),
        client_factory=lambda config: _S3Client(),
        resolver=_private_resolver,
    )
    kwargs = {
        "job_id": "job",
        "mode": "structured",
        "body": b"{}",
        "content_type": "application/json",
        "sha256": "0" * 64,
        "presign_ttl_seconds": 0,
    }
    with pytest.raises(WorkerError) as caught:
        uploader.upload(
            config=S3Config("bucket-name", "https://private.example", "us-east-1"),
            **kwargs,
        )
    assert caught.value.code == "SOURCE_URL_FORBIDDEN"

    with pytest.raises(WorkerError) as caught:
        uploader.upload(config=S3Config("x", None, "us-east-1"), **kwargs)
    assert caught.value.code == "INVALID_S3_CONFIG"


def test_custom_s3_endpoint_requires_an_exact_allowlist() -> None:
    kwargs = {
        "config": S3Config("bucket-name", "https://s3.example.com", "us-east-1"),
        "job_id": "job",
        "mode": "structured",
        "body": b"{}",
        "content_type": "application/json",
        "sha256": "0" * 64,
        "presign_ttl_seconds": 0,
    }
    for allowlist in ((), ("*.example.com",)):
        uploader = S3Uploader(
            Settings(s3_endpoint_host_allowlist=allowlist),
            client_factory=lambda config: _S3Client(),
            resolver=_public_resolver,
        )
        with pytest.raises(WorkerError) as caught:
            uploader.upload(**kwargs)
        assert caught.value.code == "INVALID_S3_CONFIG"


def test_unknown_s3_client_failures_are_not_retryable_and_are_sanitized() -> None:
    uploader = S3Uploader(Settings(), client_factory=lambda config: _S3Client(fail=True))
    with pytest.raises(WorkerError) as caught:
        uploader.upload(
            config=S3Config("bucket-name", None, "us-east-1"),
            job_id="job",
            mode="structured",
            body=b"{}",
            content_type="application/json",
            sha256="0" * 64,
            presign_ttl_seconds=0,
        )
    assert caught.value.code == "S3_UPLOAD_FAILED"
    assert caught.value.retryable is False
    assert "credentials" not in caught.value.message


def test_permanent_s3_client_errors_are_not_retryable() -> None:
    class PermanentFailure(_S3Client):
        def put_object(self, **kwargs):
            del kwargs
            raise ClientError(
                {
                    "Error": {"Code": "AccessDenied", "Message": "secret details"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "PutObject",
            )

    uploader = S3Uploader(Settings(), client_factory=lambda config: PermanentFailure())
    with pytest.raises(WorkerError) as caught:
        uploader.upload(
            config=S3Config("bucket-name", None, "us-east-1"),
            job_id="job",
            mode="structured",
            body=b"{}",
            content_type="application/json",
            sha256="0" * 64,
            presign_ttl_seconds=0,
        )
    assert caught.value.code == "S3_UPLOAD_FAILED"
    assert caught.value.retryable is False
    assert S3Uploader._is_retryable_error(NoCredentialsError()) is False
    assert (
        S3Uploader._is_retryable_error(
            EndpointConnectionError(endpoint_url="https://s3.example.com")
        )
        is True
    )


def test_custom_s3_transport_connects_to_the_validated_address(monkeypatch) -> None:
    resolutions = 0

    def resolver(*args, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return _public_resolver(*args, **kwargs)

    uploader = S3Uploader(
        Settings(s3_endpoint_host_allowlist=("s3.example.com",)),
        resolver=resolver,
    )
    sent = {}

    def send_pinned(**kwargs):
        sent.update(kwargs)
        return 200, {"etag": '"pinned-etag"'}, b""

    monkeypatch.setattr(uploader, "_send_pinned", send_pinned)
    body = b"{}"
    digest = hashlib.sha256(body).hexdigest()
    result = uploader.upload(
        config=S3Config(
            "bucket-name",
            "https://s3.example.com",
            "us-east-1",
            "access",
            "secret",
        ),
        job_id="job",
        mode="structured",
        body=body,
        content_type="application/json",
        sha256=digest,
        presign_ttl_seconds=60,
    )

    assert resolutions == 1
    assert sent["host"] == "s3.example.com"
    assert sent["address"] == "93.184.216.34"
    assert sent["url"].startswith("https://s3.example.com/bucket-name/nuextract3/")
    assert sent["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert result["etag"] == "pinned-etag"
    assert result["url"].startswith(sent["url"] + "?X-Amz-Algorithm=AWS4-HMAC-SHA256")


def test_standard_s3_client_ignores_configured_endpoint_overrides(monkeypatch) -> None:
    captured = {}

    def client(service_name, **kwargs):
        captured["service_name"] = service_name
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://127.0.0.1:9000")
    monkeypatch.setattr(boto3, "client", client)
    result = S3Uploader._client(S3Config("bucket-name", None, "us-east-1"))

    assert result is not None
    assert captured["service_name"] == "s3"
    assert captured["endpoint_url"] is None
    assert captured["config"].ignore_configured_endpoint_urls is True


def test_custom_s3_rejects_malformed_urls_and_retries_timeout_codes(
    monkeypatch,
) -> None:
    uploader = S3Uploader(
        Settings(s3_endpoint_host_allowlist=("s3.example.com",)),
        resolver=_public_resolver,
    )
    kwargs = {
        "job_id": "job",
        "mode": "structured",
        "body": b"{}",
        "content_type": "application/json",
        "sha256": "0" * 64,
        "presign_ttl_seconds": 0,
    }
    with pytest.raises(WorkerError) as caught:
        uploader.upload(
            config=S3Config("bucket-name", "https://[", "us-east-1", "key", "secret"),
            **kwargs,
        )
    assert caught.value.code == "INVALID_S3_CONFIG"
    assert caught.value.retryable is False

    monkeypatch.setattr(
        uploader,
        "_send_pinned",
        lambda **kwargs: (
            400,
            {},
            b"<Error><Code>RequestTimeout</Code></Error>",
        ),
    )
    with pytest.raises(WorkerError) as caught:
        uploader.upload(
            config=S3Config(
                "bucket-name",
                "https://s3.example.com",
                "us-east-1",
                "key",
                "secret",
            ),
            **kwargs,
        )
    assert caught.value.code == "S3_UPLOAD_FAILED"
    assert caught.value.retryable is True
