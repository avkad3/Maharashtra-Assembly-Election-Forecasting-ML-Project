"""Project-wide paths, data sources and party -> alliance ("camp") mappings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Public datasets hosted on OpenCity (data.opencity.in). The 2014 and 2019
# files are in the TCPD Lok Dhaba format; the 2024 file is ECI final results.
DATA_SOURCES = {
    "mh_2014.csv": "https://data.opencity.in/dataset/58a7ac4c-88ae-43f2-bf97-8e469bc1d88f/resource/475824d0-e7d9-4c84-bcec-45c95002bed8/download/8f8325f1-a46f-41fd-b7b8-9508de3f645d.csv",
    "mh_2019.csv": "https://data.opencity.in/dataset/b083dee8-a721-4f30-af85-e0e6e59463b4/resource/74fe80ab-2436-43dc-baf9-e9cc6524494e/download/382ff87e-e728-4215-8367-2f93c29d234d.csv",
    "mh_2024.csv": "https://data.opencity.in/dataset/0e9001ca-b19d-43e0-b0f5-73ed286bbacc/resource/1b620173-b21b-4a58-8304-93095429fab5/download/a6c740cd-c7c5-4d04-a88f-3835aabbe0fe.csv",
    "mh_2024_electors.csv": "https://data.opencity.in/dataset/f88ec36b-f2fe-4bad-9819-18ae454be64b/resource/04a8e067-678f-4365-a9b9-1a5df7b34982/download/3aa46b12-eebc-4812-9025-940dd1749e86.csv",
}

ELECTION_YEARS = [2014, 2019, 2024]
TOTAL_SEATS = 288
MAJORITY = 145

# The 2024 ECI file uses full party names; normalise them to the TCPD codes
# used in 2014/2019 (with new codes for the post-2022 splinter parties).
PARTY_NAME_2024 = {
    "Bharatiya Janata Party": "BJP",
    "Shiv Sena": "SHS",
    "Nationalist Congress Party": "NCP",
    "Shiv Sena (Uddhav Balasaheb Thackeray)": "SHS_UBT",
    "Indian National Congress": "INC",
    "Nationalist Congress Party- Sharadchandra Pawar": "NCP_SP",
    "Samajwadi Party": "SP",
    "Independent": "IND",
    "Jan Surajya Shakti": "JSS",
    "Rashtriya Yuva Swabhiman Party": "RYSP",
    "Rashtriya Samaj Paksha": "RSPS",
    "All India Majlis-E-Ittehadul Muslimeen": "AIMIM",
    "Communist Party of India  (Marxist)": "CPIM",
    "Peasants And Workers Party of India": "PWPI",
    "Rajarshi Shahu Vikas Aghadi": "RSVA",
    "Vanchit Bahujan  Aaghadi": "VBA",
    "Maharashtra Navnirman sena": "MNS",
    "Bahujan Samaj Party": "BSP",
    "Bahujan Vikas Aaghadi": "BVA",
    "Prahar Janshakti Party": "PHJSP",
    "Swabhimani Paksha": "SWP",
    "None of the Above": "NOTA",
}

# Two broad camps plus "OTH". In 2014 the four big parties contested
# separately, but BJP/SHS and INC/NCP are still grouped by their usual camp so
# that "camp strength" is comparable across elections.
#   NDA  -> 2014/2019 BJP-Shiv Sena camp, 2024 Mahayuti
#   UPA  -> 2014/2019 Congress-NCP camp, 2024 Maha Vikas Aghadi (MVA)
CAMPS = {
    2014: {
        "NDA": {"BJP", "SHS", "RSPS", "RPI(A)", "SWP"},
        "UPA": {"INC", "NCP"},
    },
    2019: {
        "NDA": {"BJP", "SHS", "RSPS", "JSS", "RPI(A)"},
        "UPA": {"INC", "NCP", "SWP", "PWPI"},
    },
    2024: {
        "NDA": {"BJP", "SHS", "NCP", "JSS", "RSPS", "RYSP", "RSVA"},
        "UPA": {"INC", "SHS_UBT", "NCP_SP", "SP", "CPIM", "PWPI"},
    },
}

# Parties whose individual vote share is used as a feature (after mapping the
# splinter parties back to their parent label where that keeps a stable meaning).
FEATURE_PARTIES = ["BJP", "SHS", "INC", "NCP", "VBA", "MNS", "AIMIM", "BSP", "IND", "NOTA"]
FEATURE_PARENT = {
    "SHS_UBT": "SHS",  # both Shiv Sena factions -> Shiv Sena voter base
    "NCP_SP": "NCP",   # both NCP factions -> NCP voter base
    "BBM": "VBA",      # Bharipa Bahujan Mahasangh merged into VBA in 2019
}

CAMP_LABELS = ["NDA", "UPA", "OTH"]
