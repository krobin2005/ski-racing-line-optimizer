"""Fetch the Gran Risa terrain and piste geometry (docs/ROADMAP_3D.md) into data/gran_risa/.

    python scripts/fetch_gran_risa.py

- Terrain: Autonomous Province of Bolzano - South Tyrol, DigitalTerrainModel-2.5m (CC0), via its
  WCS service, in ETRS89 / UTM 32N (EPSG:25832). The 0.5 m LiDAR model only covers settlement areas
  and has no data on the slope.
- Piste: OpenStreetMap ways named "Gran Risa" with piste:type=downhill (centreline and outline),
  via the Overpass API. (c) OpenStreetMap contributors, ODbL.

Writes data/gran_risa/terrain.npz (elevation grid + georeference) and osm_pistes.json.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "gran_risa"
WCS = "https://geoservices9.civis.bz.it/geoserver/p_bz-Elevation/ows"
COVERAGE = "p_bz-Elevation__DigitalTerrainModel-2.5m"
BBOX_UTM = (721800, 723400, 5160500, 5163100)  # E min, E max, N min, N max (EPSG:25832)
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = ('[out:json][timeout:60];way["piste:type"="downhill"]["name"~"Risa",i]'
                  "(46.55,11.85,46.62,11.95);out geom tags;")
USER_AGENT = "ski-racing-line-optimizer (student research project)"


def _get(url: str, data: bytes | None = None, attempts: int = 4) -> bytes:
    """GET (or POST with data), retrying with backoff: public Overpass servers often return 502/504
    under load."""
    import time

    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 502, 503, 504) or attempt == attempts - 1:
                raise
            time.sleep(10 * 2**attempt)
    raise RuntimeError("unreachable")


def parse_arcgrid(text: str):
    """ArcGrid ASCII -> (z with NaN for no-data, row 0 = north, x of west cell edge, y of south edge, cell)."""
    lines = text.splitlines()
    header = {}
    for line in lines[:6]:
        key, value = line.split()
        header[key.lower()] = float(value)
    z = np.loadtxt(io.StringIO("\n".join(lines[6:])), dtype=np.float32)
    z[z == header["nodata_value"]] = np.nan
    return z, header["xllcorner"], header["yllcorner"], header["cellsize"]


def fetch_terrain() -> Path:
    e0, e1, n0, n1 = BBOX_UTM
    query = urllib.parse.urlencode({
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage", "coverageId": COVERAGE,
        "format": "text/plain",
    }) + f"&subset=E({e0},{e1})&subset=N({n0},{n1})"
    z, x0, y0, cell = parse_arcgrid(_get(f"{WCS}?{query}").decode())
    # Store with row 0 = south, so z[i, j] is at (x0 + (j + 0.5) cell, y0 + (i + 0.5) cell).
    z = z[::-1].copy()
    path = OUT / "terrain.npz"
    np.savez_compressed(path, z=z, x0=x0, y0=y0, cell=cell, crs="EPSG:25832", source=COVERAGE)
    return path


def fetch_piste() -> Path:
    raw = _get(OVERPASS, urllib.parse.urlencode({"data": OVERPASS_QUERY}).encode())
    path = OUT / "osm_pistes.json"
    path.write_text(json.dumps(json.loads(raw), indent=1))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (fetch_terrain(), fetch_piste()):
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
