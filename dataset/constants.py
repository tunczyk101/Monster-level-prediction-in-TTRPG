import os


RANDOM_STATE = 42

PATHFINDER_BLOG_SOURCE = "Pathfinder Blog"

ORIGINAL_DATA_FOLDER = os.path.join("..", "pf2e", "packs", "pf2e")
SAVE_DIR = os.path.join("dataset", "pathfinder_2e_remaster_data_2026")
PROCESSED_BESTIARIES_FOLDER = "preprocessed_bestiaries"
BOOKS_WITH_DATES = os.path.join("dataset", "books_with_dates.csv")

FEATURES = [
    "cha",
    "con",
    "dex",
    "int",
    "str",
    "wis",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "land_speed",
    "num_immunities",
    "fly",
    "swim",
    "climb",
    "melee",
    "ranged",
    "spells",
    "spell_dc",
    "max_spell",
]

ORDERED_CHARACTERISTICS_BASIC = ["str", "dex", "con", "int", "wis", "cha", "ac", "hp"]
ORDERED_CHARACTERISTICS_EXPANDED = [
    "str",
    "dex",
    "con",
    "int",
    "wis",
    "cha",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "melee_max_bonus",
    "avg_melee_dmg",
    "ranged_max_bonus",
    "avg_ranged_dmg",
    "max_spell_lvl",
    "spell_dc",
    "spell_attack",
]
ORDERED_CHARACTERISTICS_FULL = [
    "str",
    "dex",
    "con",
    "int",
    "wis",
    "cha",
    "ac",
    "hp",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "perception",
    "num_immunities",
    "land_speed",
    "fly",
    "swim",
    "spells_nr_lvl_1",
    "spells_nr_lvl_2",
    "spells_nr_lvl_3",
    "spells_nr_lvl_4",
    "spells_nr_lvl_5",
    "spells_nr_lvl_6",
    "spells_nr_lvl_7",
    "spells_nr_lvl_8",
    "spells_nr_lvl_9",
    "melee_max_bonus",
    "avg_melee_dmg",
    "ranged_max_bonus",
    "avg_ranged_dmg",
    "max_spell_lvl",
    "spell_dc",
    "spell_attack",
]
