"""Safe image intake for the app and pipeline.

* `decode_image_file` - decode an uploaded file with Pillow: format whitelist, size checks *before* pixel
  data is decoded, decompression-bomb protection, EXIF orientation, mode conversion.
* `prepare_array`     - validate an in-memory array (shape, channels, dtype, finite values, pixel budget)
  and downscale it in its native dtype *before* converting to float32, so a huge upload never allocates a
  huge float array.

Every rejection raises ImageValidationError (a ValueError) with a message suitable for end users.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import cv2
import numpy as np

from lumen.utils import ensure_rgb, to_float

MIN_SIDE = 16
DEFAULT_MAX_PIXELS = int(float(os.environ.get("LUMEN_MAX_PIXELS", 40e6)))      # 40 MP
DEFAULT_MAX_FILE_BYTES = int(float(os.environ.get("LUMEN_MAX_FILE_MB", 20)) * 2**20)
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO"}  # MPO = multi-picture JPEG from phones
SUPPORTED_TEXT = "JPEG, PNG, WebP, BMP or TIFF"


class ImageValidationError(ValueError):
    """The input is not an image LUMEN can process; the message is shown to the user."""


def _check_dims(w: int, h: int, max_pixels: int) -> None:
    if min(w, h) < MIN_SIDE:
        raise ImageValidationError(f"Image is too small ({w}x{h}); each side must be at least {MIN_SIDE} px.")
    if w * h > max_pixels:
        raise ImageValidationError(
            f"Image is too large ({w}x{h} = {w * h / 1e6:.1f} MP); the limit is {max_pixels / 1e6:.0f} MP. "
            "Please resize it and try again."
        )


def prepare_array(img, max_side: int = 0, max_pixels: int = DEFAULT_MAX_PIXELS) -> np.ndarray:
    """Validate an image array and return float32 RGB in [0, 1], longest side <= max_side (0 = no limit)."""
    if img is None:
        raise ImageValidationError("No image provided.")
    a = np.asarray(img)
    if a.dtype == object or a.size == 0:
        raise ImageValidationError("The image is empty or not a numeric array.")
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)):
        raise ImageValidationError(
            f"Unsupported image shape {a.shape}; expected HxW (gray), HxWx3 (RGB) or HxWx4 (RGBA), "
            f"each side >= {MIN_SIDE} px."
        )
    h, w = a.shape[:2]
    _check_dims(w, h, max_pixels)  # before any conversion or allocation

    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype.kind == "f":
        if not np.isfinite(a).all():
            raise ImageValidationError("The image contains NaN or infinite values.")
        a = a.astype(np.float32, copy=False)
    elif a.dtype.kind in "iu" and a.dtype not in (np.uint8, np.uint16):
        lo, hi = int(a.min()), int(a.max())
        if lo < 0 or hi > 65535:
            raise ImageValidationError(f"Integer pixel values must lie in [0, 65535], got [{lo}, {hi}].")
        a = a.astype(np.uint8 if hi <= 255 else np.uint16)
    elif a.dtype.kind not in "iuf":
        raise ImageValidationError(f"Unsupported pixel type {a.dtype}.")

    if max_side and max(h, w) > max_side:  # downscale in the native dtype (uint8: 1 byte/channel)
        k = max_side / max(h, w)
        a = cv2.resize(a, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return to_float(ensure_rgb(a))


def _to_array(im) -> np.ndarray:
    """Pillow image -> uint8/uint16/float32 numpy array (gray or RGB)."""
    if im.mode.startswith("I;16"):
        return np.asarray(im, dtype=np.uint16)
    if im.mode == "I":
        a = np.asarray(im)
        if a.min() < 0 or a.max() > 65535:
            raise ImageValidationError("32-bit integer images outside the 16-bit range are not supported.")
        return a.astype(np.uint16)
    if im.mode == "F":
        a = np.asarray(im, dtype=np.float32)
        if not np.isfinite(a).all():
            raise ImageValidationError("The image contains NaN or infinite values.")
        m = float(a.max())
        return np.clip(a / m if m > 1 else a, 0, 1)
    return np.asarray(im.convert("RGB"))


def decode_image_file(path, max_pixels: int = DEFAULT_MAX_PIXELS,
                      max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> np.ndarray:
    """Decode an uploaded image file safely. Returns a numpy array (not yet resized / float)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    p = Path(path)
    if not p.is_file():
        raise ImageValidationError("The uploaded file could not be found - please upload it again.")
    size = p.stat().st_size
    if size == 0:
        raise ImageValidationError("The uploaded file is empty.")
    if size > max_bytes:
        raise ImageValidationError(f"The file is {size / 2**20:.1f} MB; the limit is {max_bytes / 2**20:.0f} MB.")

    with warnings.catch_warnings():
        # Pillow warns above Image.MAX_IMAGE_PIXELS and raises above twice that; treat both as errors
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(p) as im:  # lazy: reads the header only
                if im.format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(
                        f"Unsupported image format '{im.format}'. Please upload {SUPPORTED_TEXT}.")
                _check_dims(*im.size, max_pixels)  # before decoding any pixel data
                im.seek(0)  # first frame of multi-frame files
                im = ImageOps.exif_transpose(im)  # phone photos: apply the orientation tag
                return _to_array(im)
        except ImageValidationError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
            raise ImageValidationError(
                f"Image rejected: too many pixels (possible decompression bomb). Limit {max_pixels / 1e6:.0f} MP."
            ) from e
        except UnidentifiedImageError as e:
            raise ImageValidationError(
                f"This file is not a recognised image. Please upload {SUPPORTED_TEXT} "
                "(HEIC/HEIF and camera RAW files are not supported - export them as JPEG or PNG first)."
            ) from e
        except (OSError, SyntaxError, ValueError) as e:
            raise ImageValidationError(f"The image file appears to be corrupted or truncated ({e}).") from e
