"""Assign feeder buses to real county boundaries or an explicit county table.

Coordinates must be WGS84 longitude/latitude, in that order. County metadata
describes an administrative area; predictions only cover the modeled feeder
buses in that area, never all customers or all feeders in the county.
"""

import csv
import io
import json
import math
from pathlib import Path
import zipfile


EARTH_RADIUS_KM = 6371.0088
SCOPE_NOTE = (
    "County risk refers only to represented feeder buses in the current operating "
    "scenario, not the entire administrative county or a future time interval."
)


def _bus_name(value):
    return str(value).strip().strip('"\'').lower().split(".", 1)[0]


def _coordinate(value, label):
    if value is None:
        return None
    try:
        longitude, latitude = float(value[0]), float(value[1])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"Invalid longitude/latitude for {label}: {value!r}") from exc
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        return None
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError(
            f"Coordinates for {label} are not WGS84 longitude/latitude: {value!r}"
        )
    return longitude, latitude


def _rows(stream):
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        raise ValueError("The CSV has no header.")
    for row in reader:
        yield {
            str(key).strip().lower(): (value or "").strip()
            for key, value in row.items()
            if key is not None
        }


def _put_coordinate(result, bus, coordinate):
    if not bus:
        raise ValueError("A bus coordinate row has an empty bus name.")
    if bus in result and result[bus] != coordinate:
        raise ValueError(f"Conflicting coordinates for bus {bus!r}.")
    result[bus] = coordinate


def load_bus_coordinates(csv_or_zip):
    """Load Busname,X,Y or bus,longitude,latitude from CSV or an IEEE ZIP.

    OpenDSS names are case-insensitive; phase suffixes are removed. Repeated
    identical coordinates are accepted, and contradictory duplicates fail.
    ZIP files are read in place without extracting any members.
    """
    path = Path(csv_or_zip).expanduser()
    result = {}

    def read(stream):
        for row in _rows(stream):
            bus = _bus_name(row.get("busname", row.get("bus", "")))
            x = row.get("longitude", row.get("lon", row.get("x")))
            y = row.get("latitude", row.get("lat", row.get("y")))
            if x is None or y is None:
                raise ValueError("Bus coordinate CSV requires Busname,X,Y or bus,longitude,latitude.")
            point = _coordinate((x, y), bus) if x and y else None
            _put_coordinate(result, bus, point)

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if name.lower().endswith("buscoords.csv") and not name.startswith("__MACOSX/")
            )
            if not members:
                raise ValueError(f"No *_Buscoords.csv found in {path}.")
            for name in members:
                with archive.open(name) as binary:
                    with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream:
                        read(stream)
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            read(stream)
    return result


def _bounds(points):
    points = list(points)
    if not points:
        return None
    return [
        min(point[0] for point in points), min(point[1] for point in points),
        max(point[0] for point in points), max(point[1] for point in points),
    ]


def _bbox_area(bounds):
    if bounds is None:
        return None
    west, south, east, north = bounds
    return (EARTH_RADIUS_KM ** 2 * math.radians(east - west)
            * abs(math.sin(math.radians(north)) - math.sin(math.radians(south))))


def _ring_area(ring):
    # Spherical area approximation; short county-boundary segments make this
    # appropriate for area context, rather than cadastral measurement.
    total = 0.0
    for first, second in zip(ring, ring[1:] + ring[:1]):
        delta = math.radians(second[0] - first[0])
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        total += delta * (math.sin(math.radians(first[1])) + math.sin(math.radians(second[1])))
    return abs(total) * EARTH_RADIUS_KM ** 2 / 2


def _ring_location(point, ring):
    """Return 0 outside, 1 inside, 2 on the ring boundary."""
    x, y = point
    inside = False
    for first, second in zip(ring, ring[1:] + ring[:1]):
        ax, ay = first
        bx, by = second
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        tolerance = 1e-11 * max(1.0, abs(bx - ax), abs(by - ay))
        if (abs(cross) <= tolerance and min(ax, bx) - 1e-11 <= x <= max(ax, bx) + 1e-11
                and min(ay, by) - 1e-11 <= y <= max(ay, by) + 1e-11):
            return 2
        if (ay > y) != (by > y):
            if x < (bx - ax) * (y - ay) / (by - ay) + ax:
                inside = not inside
    return int(inside)


def _contains(point, county):
    west, south, east, north = county["bounds"]
    if not (west <= point[0] <= east and south <= point[1] <= north):
        return False
    for polygon in county["polygons"]:
        outer = _ring_location(point, polygon[0])
        if not outer:
            continue
        if outer == 2:
            return True
        holes = [_ring_location(point, hole) for hole in polygon[1:]]
        if 2 in holes or not any(holes):
            return True
    return False


def _county_key(name, geoid, statefp=""):
    return f"geoid:{geoid}" if geoid else f"name:{statefp}:{name.casefold()}"


def _read_counties(county_geojson):
    if isinstance(county_geojson, dict):
        document = county_geojson
    else:
        with Path(county_geojson).expanduser().open(encoding="utf-8-sig") as stream:
            document = json.load(stream)
    if document.get("type") != "FeatureCollection":
        raise ValueError("County GeoJSON must be a WGS84 FeatureCollection.")
    crs = document.get("crs", {}).get("properties", {}).get("name", "")
    if crs and not any(value in crs.upper() for value in ("4326", "CRS84")):
        raise ValueError("County GeoJSON must use WGS84 longitude/latitude (EPSG:4326).")
    counties = {}
    for number, feature in enumerate(document.get("features", [])):
        properties = feature.get("properties") or {}
        normalized = {str(key).upper(): value for key, value in properties.items()}
        name = str(normalized.get("NAMELSAD") or normalized.get("NAME") or normalized.get("COUNTY") or "").strip()
        geoid = str(normalized.get("GEOID") or normalized.get("GEOID20") or "").strip()
        statefp = str(normalized.get("STATEFP") or normalized.get("STATEFP20") or "").strip()
        if not name:
            raise ValueError(f"County feature {number} requires NAME or NAMELSAD.")
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            coordinates = [geometry.get("coordinates", [])]
        elif geometry.get("type") == "MultiPolygon":
            coordinates = geometry.get("coordinates", [])
        else:
            raise ValueError(f"County {name!r} must have Polygon or MultiPolygon geometry.")
        polygons = []
        for polygon in coordinates:
            rings = []
            for ring in polygon:
                if len(ring) < 4:
                    raise ValueError(f"County {name!r} has a polygon ring with fewer than four positions.")
                points = [_coordinate(point, f"county {name}") for point in ring]
                if any(point is None for point in points):
                    raise ValueError(f"County {name!r} has non-finite polygon coordinates.")
                if points[0] != points[-1]:
                    raise ValueError(f"County {name!r} has an unclosed polygon ring.")
                rings.append(points)
            if not rings:
                raise ValueError(f"County {name!r} has an empty polygon.")
            polygons.append(rings)
        if not polygons:
            raise ValueError(f"County {name!r} has empty geometry.")
        key = _county_key(name, geoid, statefp)
        if key in counties:
            raise ValueError(f"Duplicate county feature {key!r}; use one MultiPolygon feature per county.")
        counties[key] = {
            "name": name, "geoid": geoid or None, "statefp": statefp or None,
            "kind": "county", "polygons": polygons, "geometry": geometry,
            "bounds": _bounds(point for polygon in polygons for ring in polygon for point in ring),
            "bounds_scope": "administrative_county",
            "area_km2": sum(max(0.0, _ring_area(polygon[0]) - sum(_ring_area(hole) for hole in polygon[1:]))
                            for polygon in polygons),
            "area_method": "spherical_polygon_approximation",
        }
    if not counties:
        raise ValueError("County GeoJSON contains no counties.")
    return counties


def _name_alias(name):
    return name.casefold().removesuffix(" county").strip()


def _read_county_csv(path, counties, coordinates):
    assignments = {}
    with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as stream:
        for row in _rows(stream):
            bus = _bus_name(row.get("bus", row.get("busname", "")))
            name = row.get("county", "").strip()
            geoid = row.get("geoid", "").strip()
            statefp = row.get("statefp", "").strip()
            if not bus or not name:
                raise ValueError("Bus county CSV requires non-empty bus and county columns.")
            key = _county_key(name, geoid, statefp)
            if key not in counties and not geoid:
                matches = [key for key, county in counties.items()
                           if _name_alias(county["name"]) == _name_alias(name)
                           and (not statefp or county["statefp"] == statefp)]
                if len(matches) > 1:
                    raise ValueError(f"County name {name!r} is ambiguous; supply GEOID in the bus county CSV.")
                if matches:
                    key = matches[0]
            if key not in counties:
                counties[key] = {
                    "name": name, "geoid": geoid or None, "statefp": statefp or None,
                    "kind": "county", "geometry": None, "bounds": None,
                    "bounds_scope": "modeled_bus_coordinates", "area_km2": None,
                    "area_method": None,
                }
            if bus in assignments and assignments[bus] != key:
                raise ValueError(f"Conflicting county assignments for bus {bus!r}.")
            assignments[bus] = key
            longitude, latitude = row.get("longitude", ""), row.get("latitude", "")
            if bus in coordinates and coordinates[bus] is None and longitude and latitude:
                coordinates[bus] = _coordinate((longitude, latitude), bus)
    return assignments


def build_geography(bus_coordinates, county_geojson=None, bus_county_csv=None, region_grid_km=None):
    """Return county assignments, represented region metadata, and coverage.

    A real boundary file or a user-supplied ``bus,county[,geoid,longitude,
    latitude]`` table is required. Unmapped buses receive a separate Unknown
    region, which is not a county. Boundary ties select the first county in
    GEOID/name order, making results independent of feature order. A CSV can
    explicitly assign buses without coordinates and overrides spatial matches.
    Only counties represented by at least one bus appear in ``regions``.
    """
    if region_grid_km is not None:
        raise ValueError("Grid cells are not counties. Supply county GeoJSON or a bus-to-county CSV.")
    if county_geojson is None and bus_county_csv is None:
        raise ValueError("County geography requires --county-geojson or --bus-county-csv; coordinates alone cannot identify counties.")
    coordinates = {}
    for bus, coordinate in bus_coordinates.items():
        normalized = _bus_name(bus)
        _put_coordinate(coordinates, normalized, _coordinate(coordinate, normalized))
    if not coordinates:
        raise ValueError("No feeder buses were provided for county assignment.")
    counties = _read_counties(county_geojson) if county_geojson is not None else {}
    assignments = _read_county_csv(bus_county_csv, counties, coordinates) if bus_county_csv else {}
    ordered_counties = sorted(counties.items())
    bus_keys = {}
    missing_coordinates = 0
    outside_counties = 0
    for bus, point in coordinates.items():
        if point is None:
            missing_coordinates += 1
        if bus in assignments:
            bus_keys[bus] = assignments[bus]
            continue
        match = None
        if point is not None:
            match = next((key for key, county in ordered_counties
                          if "polygons" in county and _contains(point, county)), None)
        if match is None and point is not None:
            outside_counties += 1
        bus_keys[bus] = match
    used = sorted({key for key in bus_keys.values() if key is not None})
    regions = []
    key_to_region = {}
    for key in used:
        metadata = {field: value for field, value in counties[key].items() if field != "polygons"}
        key_to_region[key] = len(regions)
        metadata["region_id"] = len(regions)
        served_points = [coordinates[bus] for bus, assigned in bus_keys.items()
                         if assigned == key and coordinates[bus] is not None]
        metadata["modeled_bounds"] = _bounds(served_points)
        if metadata["bounds"] is None:
            metadata["bounds"] = metadata["modeled_bounds"]
        metadata["modeled_bbox_area_km2"] = _bbox_area(metadata["modeled_bounds"])
        metadata["bus_count"] = sum(assigned == key for assigned in bus_keys.values())
        metadata["scope_note"] = SCOPE_NOTE
        regions.append(metadata)
    unassigned = sum(key is None for key in bus_keys.values())
    if unassigned:
        key_to_region[None] = len(regions)
        served_points = [coordinates[bus] for bus, assigned in bus_keys.items()
                         if assigned is None and coordinates[bus] is not None]
        modeled_bounds = _bounds(served_points)
        regions.append({
            "region_id": len(regions), "name": "Unknown", "geoid": None,
            "statefp": None, "kind": "unknown", "geometry": None,
            "bounds": modeled_bounds, "bounds_scope": "modeled_bus_coordinates",
            "modeled_bounds": modeled_bounds, "area_km2": None, "area_method": None,
            "modeled_bbox_area_km2": _bbox_area(modeled_bounds), "bus_count": unassigned,
            "scope_note": "Unassigned feeder buses; this group is not an administrative county.",
        })
    total = len(coordinates)
    return {
        "bus_to_region": {bus: key_to_region[key] for bus, key in bus_keys.items()},
        "bus_coordinates": coordinates,
        "regions": regions,
        "coverage": {
            "total_buses": total, "assigned_buses": total - unassigned,
            "unassigned_buses": unassigned, "assigned_fraction": (total - unassigned) / total,
            "missing_coordinates": missing_coordinates, "outside_counties": outside_counties,
            "county_count": len(used),
            "source": "geojson+csv" if county_geojson is not None and bus_county_csv else
                      "geojson" if county_geojson is not None else "csv",
            "scope_note": SCOPE_NOTE,
        },
    }
