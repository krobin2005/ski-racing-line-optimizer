# Gran Risa data

Fetched by `python scripts/fetch_gran_risa.py`.

| File | Contents | Source and licence |
| --- | --- | --- |
| `terrain.npz` | Elevation grid, 2.5 m cells, ETRS89 / UTM 32N (EPSG:25832), box E 721800–723400, N 5160500–5163100. `z[i, j]` is centred at (x0 + (j + 0.5)·cell, y0 + (i + 0.5)·cell). | Autonomous Province of Bolzano – South Tyrol, *DigitalTerrainModel-2.5m*, via its WCS service. Licence **CC0 1.0**. The 0.5 m LiDAR model only covers settlement areas and has no data on the slope. |
| `race_2025_run1.json` | Official data for the 21 Dec 2025 World Cup GS, run 1: technical data, snow and temperature, Marco Schwarz's split times (22.25 / 40.82 / 56.93 / 76.00 s), run-1 top 5. The positions of the 3 timing points are not published. | FIS official results and analysis PDFs (Longines timing). |
| `osm_pistes.json` | OpenStreetMap ways named "Gran Risa" with `piste:type=downhill`: the piste line (69 points) and the piste outline (closed, 139 points). | © OpenStreetMap contributors, **ODbL 1.0**, via the Overpass API. |

Checks against published course facts (tests/test_terrain3d.py):
- vertical drop 450 m between the 1,870 m start and the 1,420 m finish (published 448 m);
- slope length about 1.25 km (published 1,225–1,255 m);
- mean gradient within the published ~36%.
