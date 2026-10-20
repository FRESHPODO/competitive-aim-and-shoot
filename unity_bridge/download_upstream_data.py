"""Download the optional original 60 Hz empirical trajectories with hash verification."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "251ce79854c33bdbfe2f28aaa6669b0e58ad6445"
SHA256 = "4751b505ebd523a87ac88840962719bdc9a1cdad1f446d6d2976aa78cb874778"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "empirical" / "trajectories_60hz.h5"
    if destination.exists() and digest(destination) == SHA256:
        print("Already verified:", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".h5.download")
    url = f"https://raw.githubusercontent.com/jsyoon-k5/Aim-and-Shoot/{REVISION}/empirical/trajectories_60hz.h5"
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if digest(temporary) != SHA256:
            raise RuntimeError("Downloaded data SHA-256 does not match the pinned upstream file")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("Downloaded and verified:", destination)


if __name__ == "__main__":
    main()
