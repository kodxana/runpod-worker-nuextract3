"""Optional S3 result delivery without local fallback."""

from __future__ import annotations

import http.client
import os
import re
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

from .config import MODEL_REVISION, Settings
from .errors import WorkerError
from .media import DnsResolver, resolve_https_target

_BUCKET_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{1,253}[A-Za-z0-9]\Z")
_MAX_S3_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True)
class S3Config:
    bucket: str
    endpoint_url: str | None
    region: str
    access_key: str | None = None
    secret_key: str | None = None
    session_token: str | None = None


def resolve_s3_config(job: dict[str, Any]) -> S3Config | None:
    per_job = job.get("s3Config")
    if per_job is not None:
        if not isinstance(per_job, dict) or set(per_job) != {
            "accessId",
            "accessSecret",
            "bucketName",
            "endpointUrl",
        }:
            raise WorkerError("INVALID_S3_CONFIG", "Top-level s3Config is invalid.")
        values = [per_job[key] for key in ("accessId", "accessSecret", "bucketName", "endpointUrl")]
        if not all(isinstance(value, str) and value for value in values):
            raise WorkerError(
                "INVALID_S3_CONFIG", "Top-level s3Config values must be non-empty strings."
            )
        return S3Config(
            bucket=per_job["bucketName"],
            endpoint_url=per_job["endpointUrl"],
            region=os.environ.get("OUTPUT_S3_REGION", os.environ.get("AWS_REGION", "us-east-1")),
            access_key=per_job["accessId"],
            secret_key=per_job["accessSecret"],
        )

    bucket = os.environ.get("OUTPUT_S3_BUCKET")
    if not bucket:
        return None
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if bool(access_key) != bool(secret_key):
        raise WorkerError(
            "INVALID_S3_CONFIG", "Both AWS access key variables must be set together."
        )
    return S3Config(
        bucket=bucket,
        endpoint_url=os.environ.get("OUTPUT_S3_ENDPOINT_URL"),
        region=os.environ.get("OUTPUT_S3_REGION", os.environ.get("AWS_REGION", "us-east-1")),
        access_key=access_key,
        secret_key=secret_key,
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


class S3Uploader:
    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[[S3Config], Any] | None = None,
        resolver: DnsResolver | None = None,
    ) -> None:
        self.settings = settings
        self.client_factory = client_factory
        self.resolver = resolver

    def upload(
        self,
        *,
        config: S3Config,
        job_id: str,
        mode: str,
        body: bytes,
        content_type: str,
        sha256: str,
        presign_ttl_seconds: int,
    ) -> dict[str, Any]:
        if not _BUCKET_RE.fullmatch(config.bucket):
            raise WorkerError("INVALID_S3_CONFIG", "S3 bucket name is invalid.")

        endpoint_target: tuple[str, str] | None = None
        if config.endpoint_url:
            allowlist = self.settings.s3_endpoint_host_allowlist
            if not allowlist or any(entry.startswith("*.") for entry in allowlist):
                raise WorkerError(
                    "INVALID_S3_CONFIG",
                    "Custom S3 endpoints require an exact S3 endpoint host allowlist.",
                )
            try:
                parsed_endpoint = urlsplit(config.endpoint_url)
            except ValueError as exc:
                raise WorkerError(
                    "INVALID_S3_CONFIG", "Custom S3 endpoint URL is invalid."
                ) from exc
            if parsed_endpoint.path not in {"", "/"} or parsed_endpoint.query:
                raise WorkerError(
                    "INVALID_S3_CONFIG",
                    "Custom S3 endpoint URLs must not contain a path or query.",
                )
            kwargs: dict[str, Any] = {}
            if self.resolver is not None:
                kwargs["resolver"] = self.resolver
            endpoint_target = resolve_https_target(
                config.endpoint_url,
                allowlist,
                **kwargs,
            )

        extension = "md" if mode == "markdown" else "json"
        safe_job_id = re.sub(r"[^A-Za-z0-9_-]", "_", job_id)[:128] or "job"
        now = datetime.now(UTC)
        key = f"nuextract3/{now:%Y/%m/%d}/{safe_job_id}/{sha256[:16]}.{extension}"
        metadata = {
            "model-revision": MODEL_REVISION,
            "mode": mode,
            "sha256": sha256,
        }

        try:
            if self.client_factory is not None:
                client = self.client_factory(config)
                response = client.put_object(
                    Bucket=config.bucket,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                    Metadata=metadata,
                )
                etag = str(response.get("ETag", "")).strip('"')
                presigned_url = self._client_presigned_url(
                    client, config.bucket, key, presign_ttl_seconds
                )
            elif endpoint_target is not None:
                if not config.access_key or not config.secret_key:
                    raise WorkerError(
                        "INVALID_S3_CONFIG",
                        "Custom S3 endpoints require explicit access credentials.",
                    )
                host, address = endpoint_target
                object_url = self._custom_object_url(host, config.bucket, key)
                etag = self._put_pinned(
                    config=config,
                    object_url=object_url,
                    host=host,
                    address=address,
                    body=body,
                    content_type=content_type,
                    metadata=metadata,
                )
                presigned_url = self._custom_presigned_url(
                    config, object_url, host, presign_ttl_seconds
                )
            else:
                client = self._client(config)
                try:
                    response = client.put_object(
                        Bucket=config.bucket,
                        Key=key,
                        Body=body,
                        ContentType=content_type,
                        Metadata=metadata,
                    )
                    etag = str(response.get("ETag", "")).strip('"')
                    presigned_url = self._client_presigned_url(
                        client, config.bucket, key, presign_ttl_seconds
                    )
                finally:
                    client.close()

            result: dict[str, Any] = {
                "delivery": "s3",
                "bucket": config.bucket,
                "key": key,
                "etag": etag,
                "content_type": content_type,
                "bytes": len(body),
                "sha256": sha256,
            }
            if presigned_url is not None:
                result["url"] = presigned_url
            return result
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(
                "S3_UPLOAD_FAILED",
                "Result upload failed.",
                retryable=self._is_retryable_error(exc),
            ) from exc

    @staticmethod
    def _client_presigned_url(client: Any, bucket: str, key: str, ttl_seconds: int) -> str | None:
        if not ttl_seconds:
            return None
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl_seconds,
        )

    @staticmethod
    def _custom_object_url(host: str, bucket: str, key: str) -> str:
        authority = f"[{host}]" if ":" in host else host
        bucket_path = quote(bucket, safe="-._~")
        key_path = quote(key, safe="/-._~")
        return f"https://{authority}/{bucket_path}/{key_path}"

    def _put_pinned(
        self,
        *,
        config: S3Config,
        object_url: str,
        host: str,
        address: str,
        body: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> str:
        from botocore.auth import S3SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        host_header = f"[{host}]" if ":" in host else host
        headers = {
            "Accept-Encoding": "identity",
            "Content-Type": content_type,
            "Host": host_header,
            **{f"x-amz-meta-{key}": value for key, value in metadata.items()},
        }
        request = AWSRequest(method="PUT", url=object_url, data=body, headers=headers)
        credentials = Credentials(config.access_key, config.secret_key, config.session_token)
        S3SigV4Auth(credentials, "s3", config.region).add_auth(request)
        status, response_headers, response_body = self._send_pinned(
            host=host,
            address=address,
            method="PUT",
            url=object_url,
            headers=dict(request.headers.items()),
            body=body,
        )
        if not 200 <= status < 300:
            error_match = re.search(rb"<Code>([A-Za-z0-9]+)</Code>", response_body)
            error_code = error_match.group(1).decode("ascii") if error_match else ""
            raise WorkerError(
                "S3_UPLOAD_FAILED",
                f"S3 endpoint returned HTTP {status}.",
                retryable=status in {408, 429}
                or status >= 500
                or error_code
                in {
                    "InternalError",
                    "OperationAborted",
                    "RequestTimeout",
                    "ServiceUnavailable",
                    "SlowDown",
                },
            )
        return response_headers.get("etag", "").strip('"')

    @staticmethod
    def _custom_presigned_url(
        config: S3Config,
        object_url: str,
        host: str,
        ttl_seconds: int,
    ) -> str | None:
        if not ttl_seconds:
            return None
        from botocore.auth import S3SigV4QueryAuth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        host_header = f"[{host}]" if ":" in host else host
        request = AWSRequest(method="GET", url=object_url, headers={"Host": host_header})
        credentials = Credentials(config.access_key, config.secret_key, config.session_token)
        S3SigV4QueryAuth(
            credentials,
            "s3",
            config.region,
            expires=ttl_seconds,
        ).add_auth(request)
        return request.url

    @staticmethod
    def _send_pinned(
        *,
        host: str,
        address: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        parsed = urlsplit(url)
        request_target = parsed.path or "/"
        if parsed.query:
            request_target = f"{request_target}?{parsed.query}"

        context = ssl.create_default_context()
        raw_socket = None
        tls_socket = None
        connection = None
        try:
            raw_socket = socket.create_connection((address, 443), timeout=5)
            tls_socket = context.wrap_socket(raw_socket, server_hostname=host)
            raw_socket = None
            connection = http.client.HTTPSConnection(host, 443, timeout=30, context=context)
            connection.sock = tls_socket
            tls_socket = None
            connection.sock.settimeout(30)
            connection.request(method, request_target, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read(_MAX_S3_RESPONSE_BYTES + 1)
            return (
                response.status,
                {key.lower(): value for key, value in response.getheaders()},
                response_body,
            )
        finally:
            if connection is not None:
                connection.close()
            elif tls_socket is not None:
                tls_socket.close()
            elif raw_socket is not None:
                raw_socket.close()

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        if isinstance(error, ssl.SSLCertVerificationError):
            return False
        if isinstance(
            error,
            (
                OSError,
                ConnectTimeoutError,
                ConnectionClosedError,
                EndpointConnectionError,
                ReadTimeoutError,
            ),
        ):
            return True
        if not isinstance(error, ClientError):
            return False
        response = error.response
        status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        code = str(response.get("Error", {}).get("Code", ""))
        return (
            status in {408, 429}
            or status >= 500
            or code
            in {
                "InternalError",
                "RequestTimeout",
                "ServiceUnavailable",
                "SlowDown",
                "Throttling",
                "ThrottlingException",
            }
        )

    @staticmethod
    def _client(config: S3Config) -> Any:
        import boto3
        from botocore.config import Config

        kwargs: dict[str, Any] = {
            "region_name": config.region,
            "endpoint_url": None,
            "config": Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 3, "mode": "standard"},
                ignore_configured_endpoint_urls=True,
            ),
        }
        if config.access_key and config.secret_key:
            kwargs.update(
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                aws_session_token=config.session_token,
            )
        return boto3.client("s3", **kwargs)
