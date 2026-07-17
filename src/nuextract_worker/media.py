"""Bounded and SSRF-aware document source loading."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import math
import socket
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import (
    MAX_BASE64_BYTES,
    MAX_DOWNLOAD_SECONDS,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_PDF_PAGES,
    MAX_RAW_IMAGE_PIXELS,
    MAX_SOURCE_BYTES,
    MAX_TOTAL_IMAGE_PIXELS,
    PDF_DPI,
    Settings,
)
from .errors import WorkerError
from .schema import Request

DnsResolver = Callable[..., list[tuple[Any, Any, Any, Any, Any]]]

_IPV6_TRANSLATION_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)
_PDFIUM_LOCK = threading.Lock()


@dataclass
class ResolvedDocument:
    items: list[dict[str, Any]]
    images: list[Any]
    source_bytes: int
    pdf_pages: int
    rendered_pixels: int

    def close(self) -> None:
        for image in self.images:
            close = getattr(image, "close", None)
            if close is not None:
                close()
        self.images.clear()


def _host_allowed(host: str, allowlist: Iterable[str]) -> bool:
    entries = tuple(allowlist)
    if not entries:
        return True
    for entry in entries:
        if entry.startswith("*."):
            suffix = entry[1:]
            if host.endswith(suffix) and host != suffix[1:]:
                return True
        elif host == entry:
            return True
    return False


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        not address.is_global
        or address.is_link_local
        or address.is_loopback
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return False
    return not (
        isinstance(address, ipaddress.IPv6Address)
        and (
            address.is_site_local
            or address.ipv4_mapped is not None
            or address.sixtofour is not None
            or address.teredo is not None
            or any(address in network for network in _IPV6_TRANSLATION_NETWORKS)
        )
    )


def resolve_https_target(
    url: str,
    allowlist: Iterable[str] = (),
    resolver: DnsResolver = socket.getaddrinfo,
) -> tuple[str, str]:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if host is None:
            raise ValueError
        ascii_host = host.encode("idna").decode("ascii").lower().rstrip(".")
        if not ascii_host:
            raise ValueError
    except (UnicodeError, ValueError) as exc:
        raise WorkerError("SOURCE_URL_FORBIDDEN", "Source URL is invalid.") from exc

    if parsed.scheme.lower() != "https":
        raise WorkerError("SOURCE_URL_FORBIDDEN", "Source URLs must use HTTPS.")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise WorkerError("SOURCE_URL_FORBIDDEN", "Source URL contains forbidden components.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise WorkerError("SOURCE_URL_FORBIDDEN", "Source URL port is invalid.") from exc
    if port not in (None, 443):
        raise WorkerError("SOURCE_URL_FORBIDDEN", "Source URLs may only use port 443.")
    if not _host_allowed(ascii_host, allowlist):
        raise WorkerError("SOURCE_URL_FORBIDDEN", "Source URL host is not allowlisted.")

    try:
        answers = resolver(ascii_host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise WorkerError(
            "SOURCE_DNS_FAILED", "Source URL host could not be resolved.", retryable=True
        ) from exc
    addresses = list(
        dict.fromkeys(answer[4][0].split("%", 1)[0] for answer in answers if answer[4])
    )
    if not addresses:
        raise WorkerError("SOURCE_DNS_FAILED", "Source URL host has no addresses.", retryable=True)
    parsed_addresses = []
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
            if not _is_public_address(parsed_address):
                raise WorkerError(
                    "SOURCE_URL_FORBIDDEN", "Source URL resolves to a non-public address."
                )
            parsed_addresses.append(parsed_address)
        except ValueError as exc:
            raise WorkerError(
                "SOURCE_DNS_FAILED", "Source URL resolved to an invalid address."
            ) from exc
    parsed_addresses.sort(key=lambda address: address.version)
    return ascii_host, str(parsed_addresses[0])


def validate_https_url(
    url: str,
    allowlist: Iterable[str] = (),
    resolver: DnsResolver = socket.getaddrinfo,
) -> str:
    host, _ = resolve_https_target(url, allowlist, resolver)
    return host


class UrlFetcher:
    def __init__(
        self,
        settings: Settings,
        *,
        resolver: DnsResolver = socket.getaddrinfo,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.resolver = resolver
        self.client_factory = client_factory

    def fetch(self, url: str, media_type: str, destination: Path, byte_limit: int) -> int:
        deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
        host, address = resolve_https_target(
            url,
            self.settings.source_host_allowlist,
            self.resolver,
        )
        parsed = urlsplit(url)
        address_authority = (
            f"[{address}]" if ipaddress.ip_address(address).version == 6 else address
        )
        pinned_url = parsed._replace(netloc=address_authority).geturl()
        host_header = f"[{host}]" if ":" in host else host
        try:
            client = self.client_factory() if self.client_factory else self._httpx_client()
            with (
                client,
                client.stream(
                    "GET",
                    pinned_url,
                    headers={
                        "Accept-Encoding": "identity",
                        "Host": host_header,
                        "User-Agent": "runpod-worker-nuextract3/0.1",
                    },
                    extensions={"sni_hostname": host},
                ) as response,
            ):
                if 300 <= response.status_code < 400:
                    raise WorkerError(
                        "SOURCE_REDIRECT_FORBIDDEN", "Source URL redirects are not allowed."
                    )
                if response.status_code != 200:
                    retryable = response.status_code == 429 or response.status_code >= 500
                    raise WorkerError(
                        "SOURCE_DOWNLOAD_FAILED",
                        "Source URL returned an unsuccessful status.",
                        retryable=retryable,
                    )
                self._validate_headers(response.headers, media_type, byte_limit)
                size = 0
                with destination.open("wb") as output:
                    for chunk in response.iter_bytes(64 * 1024):
                        if time.monotonic() > deadline:
                            raise WorkerError(
                                "SOURCE_DOWNLOAD_TIMEOUT",
                                "Source download exceeded its deadline.",
                                retryable=True,
                            )
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > byte_limit:
                            raise WorkerError(
                                "SOURCE_TOO_LARGE", "Downloaded source exceeds the byte limit."
                            )
                        output.write(chunk)
                if time.monotonic() > deadline:
                    raise WorkerError(
                        "SOURCE_DOWNLOAD_TIMEOUT",
                        "Source download exceeded its deadline.",
                        retryable=True,
                    )
                return size
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(
                "SOURCE_DOWNLOAD_FAILED", "Source download failed.", retryable=True
            ) from exc

    def _httpx_client(self) -> Any:
        import httpx

        return httpx.Client(
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                connect=self.settings.download_connect_timeout_seconds,
                read=self.settings.download_read_timeout_seconds,
                write=self.settings.download_read_timeout_seconds,
                pool=self.settings.download_connect_timeout_seconds,
            ),
        )

    @staticmethod
    def _validate_headers(headers: Any, media_type: str, byte_limit: int) -> None:
        encoding = (headers.get("content-encoding") or "identity").lower()
        if encoding != "identity":
            raise WorkerError(
                "SOURCE_ENCODING_UNSUPPORTED", "Compressed HTTP responses are not accepted."
            )
        length = headers.get("content-length")
        if length is not None:
            try:
                parsed_length = int(length)
                if parsed_length < 0:
                    raise ValueError
                if parsed_length > byte_limit:
                    raise WorkerError(
                        "SOURCE_TOO_LARGE", "Source Content-Length exceeds the byte limit."
                    )
            except ValueError as exc:
                raise WorkerError(
                    "SOURCE_DOWNLOAD_FAILED", "Source Content-Length is invalid."
                ) from exc
        received_type = (headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if received_type and received_type not in {media_type, "application/octet-stream"}:
            raise WorkerError(
                "MEDIA_TYPE_MISMATCH", "Source Content-Type does not match media_type."
            )


class MediaResolver:
    def __init__(
        self,
        settings: Settings,
        *,
        url_fetcher: UrlFetcher | None = None,
    ) -> None:
        self.settings = settings
        self.url_fetcher = url_fetcher or UrlFetcher(settings)

    def resolve(self, request: Request, temp_dir: str) -> ResolvedDocument:
        items: list[dict[str, Any]] = []
        images: list[Any] = []
        source_bytes = 0
        pdf_pages = 0
        rendered_pixels = 0
        root = Path(temp_dir)

        try:
            for index, source in enumerate(request.sources):
                if source["type"] == "text":
                    items.append({"type": "text", "text": source["text"]})
                    continue

                path = root / f"source-{index}.bin"
                remaining = MAX_SOURCE_BYTES - source_bytes
                if remaining <= 0:
                    raise WorkerError("SOURCE_TOO_LARGE", "Combined sources exceed the byte limit.")
                if source["type"] == "base64":
                    size = self._decode_base64(
                        source["data"], path, min(remaining, MAX_BASE64_BYTES)
                    )
                else:
                    size = self.url_fetcher.fetch(
                        source["url"], source["media_type"], path, remaining
                    )
                source_bytes += size
                self._verify_magic(path, source["media_type"])

                if source["media_type"] == "application/pdf":
                    pages = self._render_pdf(
                        path,
                        source.get("pages"),
                        MAX_PDF_PAGES - pdf_pages,
                    )
                    page_pixels = sum(image.width * image.height for image in pages)
                    if rendered_pixels + page_pixels > MAX_TOTAL_IMAGE_PIXELS:
                        for image in pages:
                            image.close()
                        raise WorkerError(
                            "PIXEL_LIMIT_EXCEEDED",
                            "Combined rendered pixels exceed the limit.",
                        )
                    pdf_pages += len(pages)
                    rendered_pixels += page_pixels
                    images.extend(pages)
                    items.extend({"type": "image", "image": image} for image in pages)
                else:
                    image = self._load_image(path, source["media_type"])
                    rendered_pixels += image.width * image.height
                    if rendered_pixels > MAX_TOTAL_IMAGE_PIXELS:
                        image.close()
                        raise WorkerError(
                            "PIXEL_LIMIT_EXCEEDED", "Combined image pixels exceed the limit."
                        )
                    images.append(image)
                    items.append({"type": "image", "image": image})

            return ResolvedDocument(items, images, source_bytes, pdf_pages, rendered_pixels)
        except Exception:
            for image in images:
                image.close()
            raise

    @staticmethod
    def _decode_base64(data: str, destination: Path, limit: int) -> int:
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorkerError("INVALID_BASE64", "Base64 source is not strictly encoded.") from exc
        if len(decoded) > limit:
            raise WorkerError("SOURCE_TOO_LARGE", "Base64 source exceeds the byte limit.")
        destination.write_bytes(decoded)
        return len(decoded)

    @staticmethod
    def _verify_magic(path: Path, media_type: str) -> None:
        with path.open("rb") as source:
            prefix = source.read(16)
        valid = {
            "application/pdf": prefix.startswith(b"%PDF-"),
            "image/jpeg": prefix.startswith(b"\xff\xd8\xff"),
            "image/png": prefix.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/webp": prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP",
        }[media_type]
        if not valid:
            raise WorkerError("MEDIA_TYPE_MISMATCH", "Source bytes do not match media_type.")

    @staticmethod
    def _load_image(path: Path, media_type: str) -> Any:
        from PIL import Image, ImageOps

        Image.MAX_IMAGE_PIXELS = MAX_RAW_IMAGE_PIXELS
        expected_format = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[
            media_type
        ]
        image: Any = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as probe:
                    if probe.format != expected_format:
                        raise WorkerError(
                            "MEDIA_TYPE_MISMATCH", "Decoded image format does not match media_type."
                        )
                    if getattr(probe, "n_frames", 1) != 1:
                        raise WorkerError("INVALID_IMAGE", "Animated images are not supported.")
                    width, height = probe.size
                    if (
                        width < 1
                        or height < 1
                        or width > MAX_IMAGE_DIMENSION
                        or height > MAX_IMAGE_DIMENSION
                    ):
                        raise WorkerError(
                            "INVALID_IMAGE", "Image dimensions are outside the allowed range."
                        )
                    if width * height > MAX_RAW_IMAGE_PIXELS:
                        raise WorkerError(
                            "PIXEL_LIMIT_EXCEEDED", "Raw image exceeds the pixel limit."
                        )
                    probe.verify()

                with Image.open(path) as source:
                    oriented = ImageOps.exif_transpose(source)
                    try:
                        if "A" in oriented.getbands() or "transparency" in oriented.info:
                            with (
                                oriented.convert("RGBA") as rgba,
                                Image.new("RGBA", rgba.size, "white") as background,
                            ):
                                background.alpha_composite(rgba)
                                image = background.convert("RGB")
                        else:
                            image = oriented.convert("RGB")
                        image.load()
                    finally:
                        if oriented is not source:
                            oriented.close()
        except WorkerError:
            if image is not None:
                image.close()
            raise
        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            if image is not None:
                image.close()
            raise WorkerError("INVALID_IMAGE", "Image could not be decoded safely.") from exc

        pixels = image.width * image.height
        if pixels > MAX_IMAGE_PIXELS:
            scale = math.sqrt(MAX_IMAGE_PIXELS / pixels)
            size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            resized = image.resize(size, Image.Resampling.LANCZOS)
            image.close()
            image = resized
        return image

    @staticmethod
    def _render_pdf(
        path: Path,
        ranges: list[dict[str, int]] | None,
        max_pages: int = MAX_PDF_PAGES,
    ) -> list[Any]:
        with _PDFIUM_LOCK:
            return MediaResolver._render_pdf_locked(path, ranges, max_pages)

    @staticmethod
    def _render_pdf_locked(
        path: Path,
        ranges: list[dict[str, int]] | None,
        max_pages: int,
    ) -> list[Any]:
        import pypdfium2 as pdfium
        from pypdfium2 import raw as pdfium_c

        images: list[Any] = []
        try:
            try:
                document = pdfium.PdfDocument(path)
            except pdfium.PdfiumError as exc:
                if exc.err_code == pdfium_c.FPDF_ERR_PASSWORD:
                    raise WorkerError("ENCRYPTED_PDF", "Encrypted PDFs are not supported.") from exc
                raise

            with document:
                page_count = len(document)
                if page_count < 1:
                    raise WorkerError("INVALID_PDF", "PDF contains no pages.")
                pages = MediaResolver._selected_pages(
                    page_count,
                    ranges,
                    max_pages,
                )
                for page_number in pages:
                    page = document[page_number - 1]
                    try:
                        page_width, page_height = page.get_size()
                        if (
                            not all(math.isfinite(value) for value in (page_width, page_height))
                            or page_width <= 0
                            or page_height <= 0
                        ):
                            raise WorkerError("INVALID_PDF", "PDF page dimensions are invalid.")
                        width = math.ceil(page_width * PDF_DPI / 72)
                        height = math.ceil(page_height * PDF_DPI / 72)
                        if width * height > MAX_IMAGE_PIXELS:
                            raise WorkerError(
                                "PIXEL_LIMIT_EXCEEDED",
                                "PDF page exceeds the rendered pixel limit.",
                            )
                        bitmap = page.render(scale=PDF_DPI / 72)
                        try:
                            if bitmap.width * bitmap.height > MAX_IMAGE_PIXELS:
                                raise WorkerError(
                                    "PIXEL_LIMIT_EXCEEDED",
                                    "Rendered PDF page exceeds the image limit.",
                                )
                            with bitmap.to_pil() as rendered:
                                image = rendered.convert("RGB")
                                image.load()
                            images.append(image)
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
            return images
        except WorkerError:
            for image in images:
                image.close()
            raise
        except Exception as exc:
            for image in images:
                image.close()
            raise WorkerError("INVALID_PDF", "PDF could not be decoded safely.") from exc

    @staticmethod
    def _selected_pages(
        page_count: int,
        ranges: list[dict[str, int]] | None,
        max_pages: int = MAX_PDF_PAGES,
    ) -> list[int]:
        if ranges is None:
            if page_count > max_pages:
                raise WorkerError(
                    "PDF_PAGE_LIMIT_EXCEEDED",
                    f"At most {MAX_PDF_PAGES} PDF pages may be rendered per job.",
                )
            return list(range(1, page_count + 1))
        pages: list[int] = []
        for page_range in ranges:
            if page_range["end"] > page_count:
                raise WorkerError(
                    "INVALID_PDF_PAGE_RANGE", "PDF page range exceeds the document page count."
                )
            pages.extend(range(page_range["start"], page_range["end"] + 1))
        if len(pages) > max_pages:
            raise WorkerError(
                "PDF_PAGE_LIMIT_EXCEEDED",
                f"At most {MAX_PDF_PAGES} PDF pages may be rendered per job.",
            )
        return pages
