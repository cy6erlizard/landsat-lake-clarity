"""Paths, dataset identifiers, and the predictor schema.

The predictor names are taken from the header of the published matchup CSV, not
from the paper's prose and not from the package's own data description. The
description and the data disagree; see ``SCHEMA_CORRECTIONS``. Where they
disagree, the data wins.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Overridable so the same code runs locally and on a Colab Drive mount.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LAKECLARITY_DATA", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
# Reports live under DATA_DIR, not the repo, because they are handoffs between
# notebooks (e.g. notebook 02 writes T05_target_lakes.csv, notebook 03 reads it).
# In Colab the repo is a shallow clone that is wiped between notebook runtimes, so
# anything written inside it is lost; only DATA_DIR (the Drive mount) survives. The
# report-critical figures and tables are copied into the repo at Phase 7 for the
# portfolio.
FIGURE_DIR = DATA_DIR / "reports" / "figures"
TABLE_DIR = DATA_DIR / "reports" / "tables"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURE_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Environmental Data Initiative entities
# --------------------------------------------------------------------------
PASTA = "https://pasta.lternet.edu/package/data/eml"

EDI_ENTITIES: dict[str, tuple[str, str, str]] = {
    # name: (package, revision, entity_id)
    "matchups": ("edi/1427", "1", "00d8e7d797b528839ed3c55297e62556"),
    "compiled_rs": ("edi/1427", "1", "fa73e9599b093898fb4bb597923d2714"),
    "predictions": ("edi/1427", "1", "3cb4f20440cbd7b8e828e4068d2ab734"),
    "data_description": ("edi/1427", "1", "fe4ec3a8af1e18b0fa8f9f8e451b55f4"),
    "lake_information": ("edi/854", "1", "007ca4f5ec02bb5809fc661dcfa7a903"),
    "lake_characteristics": ("edi/854", "1", "fd7fe936d290a12bc6dbf5c41047849e"),
}

# Approximate on-disk sizes, so callers can refuse to load the wrong thing.
EDI_SIZES_BYTES: dict[str, int] = {
    "matchups": 425_737_352,
    "compiled_rs": 10_118_930_682,
    "predictions": 7_546_775_561,
    "lake_information": 128_444_012,
    "lake_characteristics": 100_193_818,
}

# --------------------------------------------------------------------------
# Predictor schema
# --------------------------------------------------------------------------
# Verified against the CSV header on 2026-07-09. The package's own
# `data_description` entity is wrong in two places, and a third column pair is
# unusable for reasons the description does not mention.
SCHEMA_CORRECTIONS = {
    "GreendivSWIR2": (
        "data_description lists `GreendivSWIR2`; the CSV header has "
        "`GreendivSWIR2median`, consistent with its fourteen siblings."
    ),
    "median_colora": (
        "data_description lists `median_colora`; the CSV has no such column. "
        "The matchup table carries six in-situ variables, not seven."
    ),
    "IMAGE_QUALITY_OLI/TIRS": (
        "Null for every Landsat 5 and Landsat 7 row (those sensors have no OLI "
        "or TIRS instrument) and constant at 9.0 for every Landsat 8 row. They "
        "carry zero information and any complete-case filter that includes them "
        "silently deletes the entire pre-2013 record. Excluded from FEATURES."
    ),
}

N_COLUMNS_PUBLISHED = 54  # not the 55 the data description implies

BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
BAND_STATS = ["median", "min", "stdDev"]

BAND_COLS = [f"{b}{s}" for b in BANDS for s in BAND_STATS]  # 18

# Every pairwise ratio of the band medians, upper triangle, published names.
RATIO_COLS = [
    "BluedivGreenmedian",
    "BluedivNIRmedian",
    "BluedivRedmedian",
    "BluedivSWIR1median",
    "BluedivSWIR2median",
    "GreendivNIRmedian",
    "GreendivRedmedian",
    "GreendivSWIR1median",
    "GreendivSWIR2median",
    "NIRdivSWIR1median",
    "NIRdivSWIR2median",
    "ReddivNIRmedian",
    "ReddivSWIR1median",
    "ReddivSWIR2median",
    "SWIR1divSWIR2median",
]

INDEX_COLS = ["KIVUmedian"]

SCENE_QUALITY_COLS = [
    "CLOUD_COVER",
    "CLOUD_COVER_LAND",
    "Pixelcount",
]

# Present in the table, deliberately not modelled. See SCHEMA_CORRECTIONS.
EXCLUDED_COLS = ["IMAGE_QUALITY_OLI", "IMAGE_QUALITY_TIRS"]

FEATURES = BAND_COLS + RATIO_COLS + INDEX_COLS + SCENE_QUALITY_COLS  # 37

ID_COLS = [
    "lagoslakeid",
    "LANDSAT_ID",
    "SATELLITE",
    "SENSING_TIME",
    "WRS_PATH",
    "WRS_ROW",
    "date",
    "sample_date",
]

INSITU_COLS = [
    "median_secchi",
    "median_doc",
    "median_chl",
    "median_colort",
    "median_tss",
    "median_ntu",
]

TARGET = "median_secchi"
LOG_TARGET = "log10_secchi"
DAY_DIFF = "Day.diff"

# The three band medians whose ratios carry the CDOM signal in stained lakes.
CDOM_RATIOS = ["BluedivRedmedian", "BluedivGreenmedian", "KIVUmedian"]

# --------------------------------------------------------------------------
# Filtering rules
# --------------------------------------------------------------------------
# LAGOS-US LANDSAT built its matchups with a +/- 7 day window, and `Day.diff` is
# close to uniform across 0..7. Piper, Glines & Rose used +/- 3, which discards
# roughly 53% of the available rows. That is a real bias/variance trade, not a
# free filter, so it is a first-class parameter of the sensitivity grid.
NATIVE_DAY_DIFF_WINDOW = 7
MAX_DAY_DIFF = 3  # per Piper, Glines & Rose (2024)
MIN_PIXELCOUNT = 10
MAX_SECCHI_M = 25.0  # physical ceiling; anything deeper is a data-entry error
MIN_SECCHI_M = 0.05

# Median band reflectance below this is a Collection 1 atmospheric-correction
# artifact, not a measurement. Dropping these rows is *not* a neutral filter.
NEGATIVE_REFLECTANCE_FLOOR = 0.0

# --------------------------------------------------------------------------
# Region and targets
# --------------------------------------------------------------------------
# Northern Lower Michigan: clear, deep, glacial kettle and moraine lakes, the
# same CDOM-influenced optical regime as the New Hampshire target lakes, with
# decades of field Secchi from the Michigan Cooperative Lakes Monitoring Program
# (CLMP, running since 1974). Chosen over the Adirondacks because interior
# Adirondack lakes have thin field records (~10 July-years) while Northern Lower
# Michigan has dozens of lakes with 20+ July-years (Glen Lake 39, Higgins 36).
# Training uses every suitable lake in the state, mirroring Piper, Glines & Rose,
# who trained on all Wisconsin lakes in AquaSat rather than only their 127 study
# lakes. Target lakes are drawn from the narrower county set below.
TRAIN_STATE = "MI"
TRAIN_REGION_NAME = "Michigan"

REGION_STATE = "MI"
REGION_NAME = "Northern Lower Michigan"
REGION_COUNTIES = [
    "Leelanau", "Grand Traverse", "Benzie", "Antrim", "Kalkaska", "Roscommon",
    "Crawford", "Otsego", "Missaukee", "Wexford", "Charlevoix", "Emmet",
    "Cheboygan", "Montmorency", "Oscoda", "Ogemaw",
]
# Coarse bounding box fallback if the county join is unusable, and the box used
# for the regional Water Quality Portal field pull.
REGION_BBOX = dict(lat_min=44.0, lat_max=45.9, lon_min=-86.3, lon_max=-83.9)

# Target-lake selection thresholds, per the plan's Phase 2 gate.
# The gate counts FIELD July-years (distinct years with a July Secchi reading in
# the Water Quality Portal), NOT coincident satellite/in-situ matchups. The
# client's validation correlates the model's July prediction (from any July pass)
# against the field July reading (from any July measurement); coincidence within
# the matchup window is a training requirement, not a validation one, so matchup
# July-years badly undercount the achievable validation sample.
MIN_FIELD_JULY_YEARS = 15
LARGE_LAKE_MIN_HA = 800.0
SMALL_LAKE_HA_RANGE = (60.0, 500.0)
# A Water Quality Portal site is assigned to the nearest plausible lake centroid.
# "Plausible" means the lake is large enough to be modelled at all and the station
# lies inside the lake's own footprint. Without the footprint test, a shoreline
# station on a large lake gets assigned to a tiny neighbouring pond whose centroid
# happens to be nearer, handing a 1 ha pond a 30-year monitoring record.
SITE_TO_LAKE_MAX_KM = 3.5      # hard cap, whatever the lake's size
MIN_MAPPABLE_LAKE_HA = 4.0     # LAGOS-US LANDSAT covers lakes over 4 ha only
SITE_SHORE_BUFFER_KM = 0.3     # tolerance beyond the equivalent-circle radius

# --------------------------------------------------------------------------
# Sensor eras. Every long time series gets these drawn on it.
# --------------------------------------------------------------------------
SENSOR_ERAS = [
    ("LANDSAT_5", 1984, 2011),
    ("LANDSAT_7", 1999, 2020),
    ("LANDSAT_8", 2013, 2020),
]

SENSOR_EVENTS = {
    2003: "L7 SLC failure",
    2011: "L5 retired",
    2013: "L8 launch",
}

RANDOM_STATE = 20260709
