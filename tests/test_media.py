from __future__ import annotations

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from PIL import Image

import nuextract_worker.media as media_module
from nuextract_worker.config import Settings
from nuextract_worker.errors import WorkerError
from nuextract_worker.media import MediaResolver, UrlFetcher, validate_https_url
from nuextract_worker.schema import validate_request


def _resolver_for(*addresses: str):
    return lambda *args, **kwargs: [(2, 1, 6, "", (address, 443)) for address in addresses]


def _png_bytes(size=(8, 8), mode="RGB") -> bytes:
    output = BytesIO()
    color = (10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30)
    with Image.new(mode, size, color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def _pdf_bytes(pages: int) -> bytes:
    import pypdfium2 as pdfium

    output = BytesIO()
    with pdfium.PdfDocument.new() as document:
        for _ in range(pages):
            page = document.new_page(width=72, height=72)
            page.close()
        document.save(output)
    return output.getvalue()


def _base64_request(data: bytes, media_type: str, pages=None):
    source = {
        "type": "base64",
        "data": base64.b64encode(data).decode(),
        "media_type": media_type,
    }
    if pages is not None:
        source["pages"] = pages
    return validate_request(
        {
            "schema_version": "1",
            "mode": "markdown",
            "sources": [source],
        }
    )


def _error_code(function, *args, **kwargs) -> str:
    with pytest.raises(WorkerError) as caught:
        function(*args, **kwargs)
    return caught.value.code


def test_https_url_accepts_public_dns_and_normalizes_host() -> None:
    host = validate_https_url(
        "https://EXAMPLE.com/document.pdf",
        resolver=_resolver_for("93.184.216.34"),
    )
    assert host == "example.com"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a",
        "https://user:pass@example.com/a",
        "https://example.com:8443/a",
        "https://example.com/a#fragment",
    ],
)
def test_https_url_rejects_forbidden_components(url: str) -> None:
    assert (
        _error_code(
            validate_https_url,
            url,
            resolver=_resolver_for("93.184.216.34"),
        )
        == "SOURCE_URL_FORBIDDEN"
    )


def test_https_url_rejects_private_or_mixed_dns_answers() -> None:
    assert (
        _error_code(
            validate_https_url,
            "https://example.com/a",
            resolver=_resolver_for("127.0.0.1"),
        )
        == "SOURCE_URL_FORBIDDEN"
    )
    for address in ("224.0.0.1", "ff02::1", "fec0::1"):
        assert (
            _error_code(
                validate_https_url,
                "https://example.com/a",
                resolver=_resolver_for(address),
            )
            == "SOURCE_URL_FORBIDDEN"
        )
    assert (
        _error_code(
            validate_https_url,
            "https://example.com/a",
            resolver=_resolver_for("64:ff9b::7f00:1"),
        )
        == "SOURCE_URL_FORBIDDEN"
    )
    assert (
        _error_code(
            validate_https_url,
            "https://example.com/a",
            resolver=_resolver_for("93.184.216.34", "10.0.0.1"),
        )
        == "SOURCE_URL_FORBIDDEN"
    )


def test_host_allowlist_supports_exact_and_subdomain_entries() -> None:
    resolver = _resolver_for("93.184.216.34")
    validate_https_url("https://files.example.com/a", ("*.example.com",), resolver)
    validate_https_url("https://example.com/a", ("example.com",), resolver)
    assert (
        _error_code(
            validate_https_url,
            "https://example.net/a",
            ("*.example.com",),
            resolver,
        )
        == "SOURCE_URL_FORBIDDEN"
    )


class _Response:
    def __init__(self, *, status=200, headers=None, chunks=()):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_bytes(self, _size):
        yield from self._chunks


class _Client:
    def __init__(self, response):
        self.response = response
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url, headers, extensions):
        self.request = (method, url, headers, extensions)
        return self.response


def test_url_fetcher_streams_a_bounded_identity_response(tmp_path) -> None:
    payload = _png_bytes()
    response = _Response(
        headers={"content-type": "image/png", "content-length": str(len(payload))},
        chunks=(payload[:10], payload[10:]),
    )
    client = _Client(response)
    fetcher = UrlFetcher(
        Settings(),
        resolver=_resolver_for("93.184.216.34"),
        client_factory=lambda: client,
    )
    destination = tmp_path / "source"

    assert fetcher.fetch("https://example.com/a.png", "image/png", destination, 1_000) == len(
        payload
    )
    assert destination.read_bytes() == payload
    assert client.request[1] == "https://93.184.216.34/a.png"
    assert client.request[2]["Accept-Encoding"] == "identity"
    assert client.request[2]["Host"] == "example.com"
    assert client.request[3] == {"sni_hostname": "example.com"}


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (_Response(status=302), "SOURCE_REDIRECT_FORBIDDEN"),
        (_Response(status=503), "SOURCE_DOWNLOAD_FAILED"),
        (_Response(headers={"content-encoding": "gzip"}), "SOURCE_ENCODING_UNSUPPORTED"),
        (_Response(headers={"content-type": "image/jpeg"}), "MEDIA_TYPE_MISMATCH"),
        (_Response(chunks=(b"123", b"456")), "SOURCE_TOO_LARGE"),
    ],
)
def test_url_fetcher_rejects_unsafe_responses(tmp_path, response, code) -> None:
    fetcher = UrlFetcher(
        Settings(),
        resolver=_resolver_for("93.184.216.34"),
        client_factory=lambda: _Client(response),
    )
    assert (
        _error_code(
            fetcher.fetch,
            "https://example.com/a.png",
            "image/png",
            tmp_path / "source",
            5,
        )
        == code
    )


def test_image_source_is_verified_loaded_and_closed(tmp_path) -> None:
    request = _base64_request(_png_bytes(mode="RGBA"), "image/png")
    document = MediaResolver(Settings()).resolve(request, str(tmp_path))

    assert len(document.images) == 1
    assert document.images[0].mode == "RGB"
    assert document.items[0]["type"] == "image"
    assert document.rendered_pixels == 64
    document.close()
    assert document.images == []


def test_image_is_resized_to_the_processed_pixel_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(media_module, "MAX_IMAGE_PIXELS", 100)
    request = _base64_request(_png_bytes(size=(20, 20)), "image/png")
    document = MediaResolver(Settings()).resolve(request, str(tmp_path))
    try:
        assert document.images[0].width * document.images[0].height <= 100
    finally:
        document.close()


def test_strict_base64_and_magic_are_enforced(tmp_path) -> None:
    invalid_base64 = validate_request(
        {
            "schema_version": "1",
            "mode": "markdown",
            "sources": [{"type": "base64", "data": "%%%%", "media_type": "image/png"}],
        }
    )
    assert _error_code(MediaResolver(Settings()).resolve, invalid_base64, str(tmp_path)) == (
        "INVALID_BASE64"
    )

    bad_magic = _base64_request(b"not a png", "image/png")
    assert _error_code(MediaResolver(Settings()).resolve, bad_magic, str(tmp_path)) == (
        "MEDIA_TYPE_MISMATCH"
    )


def test_pdf_page_ranges_are_rendered_in_order(tmp_path) -> None:
    request = _base64_request(
        _pdf_bytes(3),
        "application/pdf",
        [{"start": 1, "end": 1}, {"start": 3, "end": 3}],
    )
    document = MediaResolver(Settings()).resolve(request, str(tmp_path))
    try:
        assert document.pdf_pages == 2
        assert len(document.images) == 2
        assert [item["type"] for item in document.items] == ["image", "image"]
    finally:
        document.close()


def test_pdfium_calls_are_globally_serialized(tmp_path, monkeypatch) -> None:
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def render(*_):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return []
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(MediaResolver, "_render_pdf_locked", render)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda _: MediaResolver._render_pdf(tmp_path / "fixture.pdf", None),
                range(2),
            )
        )

    assert max_active == 1


def test_pdf_page_cap_applies_across_all_sources(tmp_path) -> None:
    encoded = base64.b64encode(_pdf_bytes(4)).decode()
    request = validate_request(
        {
            "schema_version": "1",
            "mode": "markdown",
            "sources": [
                {
                    "type": "base64",
                    "data": encoded,
                    "media_type": "application/pdf",
                    "pages": [{"start": 1, "end": 4}],
                },
                {
                    "type": "base64",
                    "data": encoded,
                    "media_type": "application/pdf",
                    "pages": [{"start": 1, "end": 3}],
                },
            ],
        }
    )
    assert _error_code(MediaResolver(Settings()).resolve, request, str(tmp_path)) == (
        "PDF_PAGE_LIMIT_EXCEEDED"
    )


def test_new_pdf_pages_are_closed_when_aggregate_pixels_overflow(tmp_path, monkeypatch) -> None:
    class TrackedImage:
        width = 10
        height = 10

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    tracked = [TrackedImage(), TrackedImage()]
    resolver = MediaResolver(Settings())
    monkeypatch.setattr(media_module, "MAX_TOTAL_IMAGE_PIXELS", 100)
    monkeypatch.setattr(resolver, "_render_pdf", lambda *args: tracked)
    request = _base64_request(b"%PDF-placeholder", "application/pdf")

    assert _error_code(resolver.resolve, request, str(tmp_path)) == "PIXEL_LIMIT_EXCEEDED"
    assert all(image.closed for image in tracked)


def test_selected_pages_reject_out_of_bounds_ranges() -> None:
    assert MediaResolver._selected_pages(5, [{"start": 2, "end": 3}]) == [2, 3]
    assert (
        _error_code(
            MediaResolver._selected_pages,
            2,
            [{"start": 1, "end": 3}],
        )
        == "INVALID_PDF_PAGE_RANGE"
    )
