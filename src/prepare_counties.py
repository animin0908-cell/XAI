"""Convert the Census county KML archive to a local GeoJSON (no GIS dependencies)."""

import argparse
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


CENSUS_URL = "https://www2.census.gov/geo/tiger/GENZ2025/kml/cb_2025_us_county_500k.zip"
NS = {"k": "http://www.opengis.net/kml/2.2"}


def convert_counties(archive, output, state="53"):
    with zipfile.ZipFile(archive) as bundle:
        names = [n for n in bundle.namelist() if n.lower().endswith(".kml")]
        if len(names) != 1:
            raise ValueError("Expected one county KML in the Census archive.")
        root = ET.fromstring(bundle.read(names[0]))
    features = []
    for place in root.findall(".//k:Placemark", NS):
        properties = {v.attrib["name"]: v.text for v in place.findall(".//k:SimpleData", NS)}
        if state and properties.get("STATEFP") != state:
            continue
        polygons = []
        for polygon in place.findall(".//k:Polygon", NS):
            rings = []
            for boundary in ["outerBoundaryIs", "innerBoundaryIs"]:
                for coords in polygon.findall(f"k:{boundary}/k:LinearRing/k:coordinates", NS):
                    rings.append([[float(v) for v in p.split(",")[:2]] for p in coords.text.split()])
            if rings:
                polygons.append(rings)
        if polygons:
            features.append({"type": "Feature", "properties": properties,
                             "geometry": {"type": "MultiPolygon", "coordinates": polygons}})
    if not features:
        raise ValueError("No counties found; check the state FIPS code and input archive.")
    result = {"type": "FeatureCollection", "source": CENSUS_URL,
              "boundary_year": 2025, "scale": "1:500,000", "features": features}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result), encoding="utf-8")
    return len(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing Census KML zip; otherwise download it.")
    parser.add_argument("--output", type=Path, default=Path("data/geography/wa_counties_2025.geojson"))
    parser.add_argument("--state", default="53", help="State FIPS; 53 = Washington, empty = all states.")
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        archive = args.output.parent / "cb_2025_us_county_500k.zip"
        if not archive.exists():
            with urllib.request.urlopen(CENSUS_URL, timeout=60) as response:
                archive.write_bytes(response.read())
    count = convert_counties(archive, args.output, args.state)
    print(f"Saved {count} counties to {args.output}")
