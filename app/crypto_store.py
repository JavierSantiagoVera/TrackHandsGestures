import io
import os
import base64

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_SALT_SIZE = 16
_ITERATIONS = 390_000


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def save_encrypted(path: str, X: np.ndarray, y: np.ndarray, password: str) -> None:
    """Compress numpy arrays and write AES-encrypted to path."""
    salt = os.urandom(_SALT_SIZE)
    key = _derive_key(password, salt)
    f = Fernet(key)

    buf = io.BytesIO()
    np.savez_compressed(buf, X=X, y=y)
    encrypted = f.encrypt(buf.getvalue())

    with open(path, "wb") as fp:
        fp.write(salt + encrypted)


def load_encrypted(path: str, password: str):
    """Return (X, y) from an encrypted file. Raises InvalidToken on wrong password."""
    with open(path, "rb") as fp:
        raw = fp.read()

    salt, encrypted = raw[:_SALT_SIZE], raw[_SALT_SIZE:]
    key = _derive_key(password, salt)
    data = Fernet(key).decrypt(encrypted)  # raises InvalidToken if wrong password

    buf = io.BytesIO(data)
    d = np.load(buf, allow_pickle=False)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)
