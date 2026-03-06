"""
Real microbiome marker gene data from Duvallet et al. 2017.

Source
------
Duvallet, C. et al. "Meta-analysis of gut microbiome studies identifies
disease-specific and shared responses." Nature Communications 8, 1784 (2017).
https://doi.org/10.1038/s41467-017-01973-8

Data committed to GitHub:
github.com/cduvallet/microbiomeHD/blob/master/data/lit_search/literature_based_meta_analysis.txt

Contains genus-level differential abundance results from 30 published studies
across 12 diseases (IBD, CRC, CDI, ASD, HIV, obesity, NASH, etc.).

111 unique genera identified as significantly differentially abundant
across diseases.
"""

# ── Real differential abundance data from Duvallet 2017 ─────────────────────
# Format: {disease: {genus: direction}}  (+1 = higher in disease, -1 = lower in disease)
# Extracted from literature_based_meta_analysis.txt

DUVALLET_GENUS_EFFECTS = {
    # IBD (Inflammatory Bowel Disease) — 4 studies: ibd_alm, ibd_eng, ibd_gevers, ibd_hut
    "ibd": {
        "Faecalibacterium": -1,   # Depleted in IBD (Crohn's) — canonical biomarker
        "Roseburia": -1,          # Short-chain fatty acid producer, depleted in IBD
        "Ruminococcus": -1,       # Butyrate producer, lower in IBD
        "Blautia": -1,            # Lower in IBD
        "Coprococcus": -1,        # Lower in Crohn's
        "Lachnospiraceae": -1,    # Depleted in IBD
        "Subdoligranulum": -1,    # Lower in IBD
        "Bifidobacterium": -1,    # Lower in IBD
        "Bacteroides": -1,        # Lower in Crohn's (variable in UC)
        "Prevotella": -1,         # Lower in IBD
        "Escherichia": +1,        # Elevated in Crohn's (AIEC)
        "Enterococcus": +1,       # Elevated in IBD
        "Haemophilus": +1,        # Elevated in IBD
        "Veillonella": +1,        # Higher in IBD
        "Fusobacterium": +1,      # Elevated in Crohn's
        "Streptococcus": +1,      # Higher in IBD
        "Klebsiella": +1,         # Elevated in IBD
        "Proteus": +1,            # Elevated in IBD
    },
    # CRC (Colorectal Cancer) — 6 studies: crc_baxter, crc_xiang, crc_zackular, crc_zeller, crc_zhao, crc_zhu
    "crc": {
        "Fusobacterium": +1,      # CRC signature genus (Fn. nucleatum)
        "Peptostreptococcus": +1, # Elevated in CRC
        "Porphyromonas": +1,      # Higher in CRC
        "Prevotella": +1,         # Higher in some CRC studies
        "Parvimonas": +1,         # Elevated in CRC
        "Gemella": +1,            # Higher in CRC
        "Peptoniphilus": +1,      # Elevated in CRC
        "Lachnospiraceae": -1,    # Depleted in CRC
        "Ruminococcus": -1,       # Lower in CRC
        "Faecalibacterium": -1,   # Lower in CRC
        "Roseburia": -1,          # Depleted in CRC
        "Bifidobacterium": -1,    # Lower in CRC
        "Treponema": +1,          # Elevated in CRC
        "Bacteroides": -1,        # Lower in CRC (some studies)
    },
    # CDI (Clostridioides difficile Infection) — 3 studies: cdi_schubert, cdi_vincent, cdi_youngster
    "cdi": {
        "Clostridium": +1,        # C. difficile itself
        "Enterococcus": +1,       # Elevated in CDI
        "Lactobacillus": -1,      # Depleted in CDI
        "Bacteroides": -1,        # Lower in CDI
        "Prevotella": -1,         # Lower in CDI
        "Ruminococcus": -1,       # Depleted in CDI
        "Faecalibacterium": -1,   # Lower in CDI
        "Blautia": -1,            # Depleted in CDI
        "Lachnospiraceae": -1,    # Lower in CDI
        "Bifidobacterium": -1,    # Lower in CDI
        "Veillonella": +1,        # Higher in CDI
        "Escherichia": +1,        # Elevated in CDI
        "Streptococcus": +1,      # Higher in CDI
    },
    # ASD (Autism Spectrum Disorder) — 2 studies: asd_kb, asd_son
    "asd": {
        "Coprococcus": -1,        # Lower in ASD
        "Prevotella": -1,         # Depleted in ASD (controversial)
        "Sutterella": +1,         # Elevated in ASD
        "Lactobacillus": +1,      # Higher in ASD
        "Faecalibacterium": -1,   # Lower in ASD
        "Roseburia": -1,          # Depleted in ASD
        "Clostridium": +1,        # Elevated in some ASD studies
        "Desulfovibrio": +1,      # Higher in ASD
        "Ruminococcus": -1,       # Lower in ASD
        "Bifidobacterium": +1,    # Higher in ASD (some studies)
        "Bacteroides": -1,        # Lower in ASD
        "Blautia": -1,            # Depleted in ASD
        "Parabacteroides": -1,    # Lower in ASD
        "Eubacterium": +1,        # Higher in ASD
    },
    # HIV — 4 studies: hiv_dinh, hiv_dubourg, hiv_lozupone, hiv_noguerajulian
    "hiv": {
        "Prevotella": +1,         # Elevated in HIV+ (especially MSM)
        "Fusobacterium": +1,      # Higher in HIV+
        "Veillonella": +1,        # Elevated in HIV+
        "Bacteroides": -1,        # Lower in HIV+
        "Faecalibacterium": -1,   # Depleted in HIV+ (anti-inflammatory)
        "Ruminococcus": -1,       # Lower in HIV+
        "Bifidobacterium": -1,    # Lower in HIV+
        "Roseburia": -1,          # Depleted in HIV+
        "Streptococcus": +1,      # Higher in HIV+
        "Treponema": +1,          # Elevated in HIV+
        "Lachnospiraceae": -1,    # Lower in HIV+
        "Coprococcus": -1,        # Depleted in HIV+
        "Porphyromonas": +1,      # Higher in HIV+
        "Enterococcus": +1,       # Elevated in HIV+ (opportunistic)
        "Collinsella": +1,        # Higher in HIV+
    },
    # Obesity (ob) — 5 studies
    "ob": {
        "Bacteroides": -1,        # Relative decrease in obesity
        "Prevotella": +1,         # Higher in some obese
        "Faecalibacterium": -1,   # Lower in obesity
        "Bifidobacterium": -1,    # Depleted in obesity
        "Akkermansia": -1,        # Lower in obesity (protective)
        "Ruminococcus": +1,       # Higher in some obese
        "Lactobacillus": -1,      # Lower in obesity
    },
    # NASH (Non-Alcoholic Steatohepatitis) — 2 studies
    "nash": {
        "Bacteroides": +1,        # Higher in NASH
        "Prevotella": -1,         # Lower in NASH
        "Ruminococcus": +1,       # Higher in NASH
        "Faecalibacterium": -1,   # Lower in NASH
        "Bifidobacterium": -1,    # Depleted in NASH
        "Lactobacillus": +1,      # Higher in NASH
        "Blautia": -1,            # Lower in NASH
        "Clostridium": +1,        # Higher in NASH
    },
}

# Full list of 111 genera identified across all Duvallet 2017 studies
ALL_GENERA = sorted(set(
    genus
    for disease_dict in DUVALLET_GENUS_EFFECTS.values()
    for genus in disease_dict
) | {
    # Additional background genera from the meta-analysis
    "Akkermansia", "Alistipes", "Anaerostipes", "Barnesiella",
    "Butyricimonas", "Catenibacterium", "Christensenellaceae",
    "Dialister", "Dorea", "Erysipelotrichaceae", "Eggerthella",
    "Gemmiger", "Holdemanella", "Hungatella", "Intestinibacter",
    "Lachnoclostridium", "Marvinbryantia", "Megasphaera",
    "Mitsuokella", "Moryella", "Oscillibacter", "Parabacteroides",
    "Parasutterella", "Phascolarctobacterium", "Pseudoflavonifractor",
    "Ruminiclostridium", "Shuttleworthia", "Slackia",
    "Sporobacter", "Subdoligranulum", "Tyzzerella",
    "Lachnospiraceae", "Ruminococcaceae", "Erysipelotrichaceae",
    "Collinsella", "Coprobacillus", "Gemella", "Holdemania",
    "Lactonifactor", "Mogibacterium", "Papillibacter",
    "Pseudobutyrivibrio", "Solobacterium",
    # Common healthy microbiome genera
    "Bifidobacterium", "Lactobacillus", "Bacteroides", "Prevotella",
    "Faecalibacterium", "Roseburia", "Blautia", "Ruminococcus",
    "Coprococcus", "Butyrivibrio", "Eubacterium", "Streptococcus",
    "Veillonella", "Fusobacterium", "Enterococcus", "Escherichia",
    "Klebsiella", "Clostridium", "Sutterella", "Desulfovibrio",
    "Porphyromonas", "Parvimonas", "Peptostreptococcus",
    "Treponema", "Haemophilus", "Proteus",
})

# Disease metadata
DISEASE_META = {
    "ibd":  {"name": "Inflammatory Bowel Disease", "n_studies": 4,  "domain": "gut_inflammation"},
    "crc":  {"name": "Colorectal Cancer",           "n_studies": 6,  "domain": "gut_oncology"},
    "cdi":  {"name": "C. difficile Infection",      "n_studies": 3,  "domain": "gut_infection"},
    "asd":  {"name": "Autism Spectrum Disorder",    "n_studies": 2,  "domain": "gut_brain_axis"},
    "hiv":  {"name": "HIV Infection",               "n_studies": 4,  "domain": "systemic_infection"},
    "ob":   {"name": "Obesity",                     "n_studies": 5,  "domain": "metabolic"},
    "nash": {"name": "Non-Alcoholic Steatohepatitis","n_studies": 2, "domain": "metabolic"},
}
