from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hub_download(repo_id: str, filename: str, revision: str, destination: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    print("Using Hugging Face Hub/Xet accelerated download.")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}.download-") as directory:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=directory,
            )
        )
        os.replace(downloaded, destination)
    return True


def download(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
) -> None:
    if destination.is_file():
        actual = sha256(destination)
        if expected_sha256 is None or actual == expected_sha256:
            print(f"Already present: {destination}")
            return
        raise RuntimeError(f"Refusing to replace {destination.name}; SHA-256 is {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}\n       to {destination}")
    partial.unlink(missing_ok=True)
    accelerated = False
    if repo_id and filename and revision:
        try:
            accelerated = _hub_download(repo_id, filename, revision, partial)
        except Exception as error:
            partial.unlink(missing_ok=True)
            print(f"Accelerated download failed ({error}); retrying with the Python downloader.")
    if not accelerated:
        print("huggingface_hub is unavailable; using the single-connection Python downloader.")
        request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Ultimate-Face-Fix/setup"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if expected_sha256 is not None:
        actual = sha256(partial)
        if actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {destination.name}: {actual}")
    os.replace(partial, destination)
