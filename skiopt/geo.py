"""Geodesy helpers for real terrain (docs/ROADMAP_3D.md): WGS84 latitude/longitude to UTM metres,
without extra dependencies. South Tyrol's terrain models are in ETRS89 / UTM zone 32N (EPSG:25832);
ETRS89 and WGS84 differ by well under a metre, which is far below the 2.5 m terrain grid.

Transverse Mercator series from Snyder (1987), "Map Projections: A Working Manual", USGS PP 1395,
eqs. 8-9 to 8-11, accurate to millimetres within a UTM zone."""

from __future__ import annotations

import numpy as np

A = 6378137.0                 # GRS80 / WGS84 semi-major axis (m)
F = 1 / 298.257222101         # GRS80 flattening
K0 = 0.9996
E2 = F * (2 - F)
EP2 = E2 / (1 - E2)


def latlon_to_utm(lat_deg, lon_deg, zone: int = 32):
    """Return (easting, northing) in metres for the northern hemisphere."""
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    lon0 = np.radians(6 * zone - 183)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = EP2 * np.cos(lat) ** 2
    a = np.cos(lat) * (lon - lon0)
    m = A * ((1 - E2 / 4 - 3 * E2**2 / 64 - 5 * E2**3 / 256) * lat
             - (3 * E2 / 8 + 3 * E2**2 / 32 + 45 * E2**3 / 1024) * np.sin(2 * lat)
             + (15 * E2**2 / 256 + 45 * E2**3 / 1024) * np.sin(4 * lat)
             - (35 * E2**3 / 3072) * np.sin(6 * lat))
    easting = K0 * n * (a + (1 - t + c) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * EP2) * a**5 / 120) + 500000.0
    northing = K0 * (m + n * np.tan(lat) * (a**2 / 2 + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
                                            + (61 - 58 * t + t**2 + 600 * c - 330 * EP2) * a**6 / 720))
    return easting, northing
