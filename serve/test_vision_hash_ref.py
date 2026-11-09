"""A picture by reference (2026-11-09): `image_url` "hash:<key>" names a picture the encoder already holds (its cache
key, the first 32 hex digits of the sha256 of the picture's bytes), so a client that re-sends its history every turn
(a computer-use agent: two screenshots a turn, 4.6 MB of them by turn 25 over a slow link) sends 70 bytes per old
picture instead of the picture.  A reference to a picture the cache does not hold is a 400 `image_not_cached` that
lists every missing key, so the client resends those as data: URLs in one go.  The engine's conversation cache keys
pictures by the same hash (#2228), so a referenced picture matches its checkpoints.

    python -m serve.test_vision_hash_ref
"""
import base64
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path

from serve.server import ImageNotCached, Vision

JPEG = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 4 + b"\xff\xd9"      # starts like a JPEG: normalize() passes it through
KEY = hashlib.sha256(JPEG).hexdigest()[:32]
DATA_URL = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()


def cache_only_vision(entries=()) -> Vision:
    """A Vision with its cache and no encoder process: what a hash: reference needs."""
    v = Vision.__new__(Vision)
    v.dir = Path(tempfile.mkdtemp())
    v.lock = threading.Lock()
    v.cache = {}
    v.cache_max = 64
    v.proc = None
    v.stopped = True
    for key in entries:
        f = v.dir / f"{key}.sve"
        f.write_bytes(b"rows")
        v.cache[key] = (f, 3)
    return v


class HashRef(unittest.TestCase):
    def test_the_key_is_the_cache_key_of_the_bytes(self):
        self.assertEqual(Vision.hash_key(f"hash:{KEY}"), KEY)
        self.assertEqual(Vision.hash_key("hash:sha256:" + hashlib.sha256(JPEG).hexdigest()), KEY)
        self.assertEqual(Vision.hash_key("HASH:" + KEY.upper()), None)      # the scheme is lower-case, as the others
        self.assertIsNone(Vision.hash_key(DATA_URL))
        self.assertIsNone(Vision.hash_key(JPEG))
        with self.assertRaises(ValueError):
            Vision.hash_key("hash:not-hex")
        with self.assertRaises(ValueError):
            Vision.hash_key("hash:" + KEY[:10])

    def test_a_cached_picture_is_answered_from_the_cache_without_an_encoder(self):
        v = cache_only_vision([KEY])
        path, n = v.encode(f"hash:{KEY}")
        self.assertEqual((path.name, n), (f"{KEY}.sve", 3))
        self.assertTrue(v.all_cached([f"hash:{KEY}", DATA_URL]))        # the bytes hash to the same key

    def test_a_reference_to_a_picture_not_held_is_image_not_cached(self):
        v = cache_only_vision([KEY])
        other = "0" * 32
        with self.assertRaises(ImageNotCached) as c:
            v.encode(f"hash:{other}")
        self.assertEqual(c.exception.keys, [other])
        self.assertIn("image_not_cached", str(c.exception))
        self.assertFalse(v.all_cached([f"hash:{other}"]))

    def test_encode_all_names_every_missing_reference_at_once(self):
        v = cache_only_vision([KEY])
        a, b = "1" * 32, "2" * 32
        with self.assertRaises(ImageNotCached) as c:
            v.encode_all([f"hash:{KEY}", f"hash:{a}", DATA_URL, f"hash:{b}", f"hash:{a}"])
        self.assertEqual(c.exception.keys, [a, b])                      # in order, each once, the held one left out

    def test_load_does_not_read_a_reference(self):
        with self.assertRaises(ValueError):
            Vision.load(f"hash:{KEY}")


if __name__ == "__main__":
    unittest.main()
