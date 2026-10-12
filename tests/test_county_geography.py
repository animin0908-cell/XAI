"""County boundaries must preserve administrative scope and unassigned buses."""

from pathlib import Path
import tempfile
import unittest
import zipfile

from src.utils.county_geography import build_geography, load_bus_coordinates


def square(west, south, east, north):
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def feature(name, geoid, polygons):
    return {
        "type": "Feature",
        "properties": {"NAME": name, "GEOID": geoid, "STATEFP": "53"},
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
    }


def collection(*features):
    return {"type": "FeatureCollection", "features": list(features)}


class CountyGeographyTests(unittest.TestCase):
    def test_polygon_holes_multipolygon_and_unknown(self):
        boundaries = collection(feature("Example", "53001", [
            [square(-120, 45, -119, 46), square(-119.8, 45.2, -119.6, 45.4)],
            [square(-118, 45, -117.8, 45.2)],
        ]))
        result = build_geography({
            "Bus_A.1.2": (-119.9, 45.1),
            "in_hole": (-119.7, 45.3),
            "on_hole": (-119.8, 45.3),
            "island": (-117.9, 45.1),
            "absent": None,
            "elsewhere": (-122, 47),
        }, county_geojson=boundaries)
        self.assertEqual(result["bus_to_region"]["bus_a"], 0)
        self.assertEqual(result["bus_to_region"]["island"], 0)
        self.assertEqual(result["bus_to_region"]["on_hole"], 0)
        self.assertEqual(result["bus_to_region"]["in_hole"], 1)
        self.assertEqual(result["regions"][1]["kind"], "unknown")
        self.assertEqual(result["coverage"]["assigned_buses"], 3)
        self.assertEqual(result["coverage"]["unassigned_buses"], 3)
        self.assertEqual(result["coverage"]["missing_coordinates"], 1)
        self.assertEqual(result["coverage"]["outside_counties"], 2)
        self.assertEqual(result["regions"][0]["bounds"], [-120.0, 45.0, -117.8, 46.0])
        self.assertIn("represented feeder buses", result["coverage"]["scope_note"])

    def test_boundary_assignment_does_not_depend_on_feature_order(self):
        left = feature("Left", "53002", [[square(-120, 45, -119, 46)]])
        right = feature("Right", "53001", [[square(-119, 45, -118, 46)]])
        for boundaries in [collection(left, right), collection(right, left)]:
            result = build_geography({"boundary": (-119, 45.5)}, boundaries)
            self.assertEqual(result["regions"][0]["geoid"], "53001")
            self.assertEqual(result["regions"][0]["bus_count"], 1)

    def test_spherical_area_accounts_for_holes(self):
        solid = collection(feature("Solid", "53001", [[square(-120, 45, -119, 46)]]))
        hollow = collection(feature("Hollow", "53001", [[
            square(-120, 45, -119, 46), square(-119.8, 45.2, -119.6, 45.4),
        ]]))
        area = build_geography({"b": (-119.9, 45.1)}, solid)["regions"][0]["area_km2"]
        hollow_area = build_geography({"b": (-119.9, 45.1)}, hollow)["regions"][0]["area_km2"]
        self.assertTrue(8500 < area < 8800)
        self.assertTrue(0 < hollow_area < area)
        self.assertEqual(build_geography({"b": (-119.9, 45.1)}, solid)["regions"][0]["modeled_bbox_area_km2"], 0)

    def test_direct_csv_assigns_missing_coordinates_and_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "county.csv"
            path.write_text("bus,county,geoid,longitude,latitude\nBUS_A.1,Franklin,53021,-119.2,46.7\nbus_b,Franklin,53021,,\n")
            result = build_geography({"bus_a": None, "bus_b": None, "bus_c": (-121, 47)}, bus_county_csv=path)
        county, unknown = result["regions"]
        self.assertEqual(county["geoid"], "53021")
        self.assertEqual(county["bus_count"], 2)
        self.assertEqual(county["bounds_scope"], "modeled_bus_coordinates")
        self.assertIsNone(county["area_km2"])
        self.assertEqual(county["modeled_bounds"], [-119.2, 46.7, -119.2, 46.7])
        self.assertEqual(result["coverage"]["missing_coordinates"], 1)
        self.assertEqual(unknown["name"], "Unknown")

    def test_conflicting_duplicate_csv_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "county.csv"
            path.write_text("bus,county,geoid\nb,A,53001\nB.1,B,53002\n")
            with self.assertRaisesRegex(ValueError, "Conflicting county assignments"):
                build_geography({"b": None}, bus_county_csv=path)

    def test_csv_and_zip_coordinate_loading_and_duplicate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coords.csv"
            path.write_text("Busname,X,Y\nBUS_A.1,-119.2,46.7\nbus_a,-119.2,46.7\nbus_b,,\n")
            expected = {"bus_a": (-119.2, 46.7), "bus_b": None}
            self.assertEqual(load_bus_coordinates(path), expected)
            archive = Path(directory) / "input.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.write(path, arcname="csv/ieee9500bal_Buscoords.csv")
                stream.writestr("../../ignored.csv", "unrelated,data\n")
            self.assertEqual(load_bus_coordinates(archive), expected)
            path.write_text("bus,longitude,latitude\nb,-119.2,46.7\nB,-120,47\n")
            with self.assertRaisesRegex(ValueError, "Conflicting coordinates"):
                load_bus_coordinates(path)

    def test_no_mapping_does_not_invent_counties(self):
        with self.assertRaisesRegex(ValueError, "coordinates alone cannot identify counties"):
            build_geography({"b": (-119, 46)})
        with self.assertRaisesRegex(ValueError, "Grid cells are not counties"):
            build_geography({"b": (-119, 46)}, region_grid_km=5)

    def test_projected_coordinates_are_rejected(self):
        boundaries = collection(feature("A", "53001", [[square(-120, 45, -119, 46)]]))
        with self.assertRaisesRegex(ValueError, "not WGS84"):
            build_geography({"b": (120000, 400000)}, boundaries)


if __name__ == "__main__":
    unittest.main()
