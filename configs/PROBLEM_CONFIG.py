species_table = dict(
    SLL="Small_leaved_lime",
    HCH="Horse_chestnut",
    LLL="Large_leaved_lime",
    CRO="Common_rowan",
    EWB="European_white_birch",
    BEE="European_beech",
    LAR="European_larch",
    SPR="Common_spruce",
    HZL="Hazel",
)
species_short = {v: k for k, v in species_table.items()}

phases_table = dict(
    LU="leaf_unfolding",
    FL="flowering",
    LC="leaf_colouring",
    NE="needle_emergence",
    NC="needle_colouring",
)
phases_short = {v: k for k, v in phases_table.items()}


climate_var_names = ["Ti", "Li", "VPDi", "ltmi", "Tmaxi", "Tmini", "Pi"]

spring_phenophases = [
    "Small_leaved_lime:leaf_unfolding",
    "Horse_chestnut:leaf_unfolding",
    "Large_leaved_lime:leaf_unfolding",
    "Common_rowan:leaf_unfolding",
    "European_white_birch:leaf_unfolding",
    "European_beech:leaf_unfolding",
    "European_larch:needle_emergence",
    "Common_spruce:needle_emergence",
    "Hazel:leaf_unfolding",
]

autumn_phenophases = [
    "Small_leaved_lime:leaf_colouring",
    "Horse_chestnut:leaf_colouring",
    "Large_leaved_lime:leaf_colouring",
    "Common_rowan:leaf_colouring",
    "European_white_birch:leaf_colouring",
    "European_beech:leaf_colouring",
    "European_larch:needle_colouring",
]

available_species_phase = [
    "Apple_tree:flowering",
    "Autumn_crocus:flowering",
    "Cherry_tree:flowering",
    "Cocksfoot_grass:flowering",
    "Coltsfoot:flowering",
    "Common_acacia:flowering",
    "Common_acacia:leaf_unfolding",
    "Common_rowan:flowering",
    "Common_rowan:leaf_colouring",
    "Common_rowan:leaf_unfolding",
    "Common_spruce:needle_emergence",
    "Cuckoo_flower:flowering",
    "Dandelion:flowering",
    "European_beech:leaf_colouring",
    "European_beech:leaf_unfolding",
    "European_elder:flowering",
    "European_larch:needle_colouring",
    "European_larch:needle_emergence",
    "European_red_elder:flowering",
    "European_white_birch:flowering",
    "European_white_birch:leaf_colouring",
    "European_white_birch:leaf_unfolding",
    "Field_daisy:flowering",
    "Grape_vine:flowering",
    "Hazel:flowering",
    "Hazel:leaf_unfolding",
    "Horse_chestnut:flowering",
    "Horse_chestnut:leaf_colouring",
    "Horse_chestnut:leaf_unfolding",
    "Large_leaved_lime:flowering",
    "Large_leaved_lime:leaf_colouring",
    "Large_leaved_lime:leaf_unfolding",
    "Pear_tree:flowering",
    "Small_leaved_lime:flowering",
    "Small_leaved_lime:leaf_colouring",
    "Small_leaved_lime:leaf_unfolding",
    "Sweet_chestnut:flowering",
    "Sweet_chestnut:leaf_colouring",
    "Sweet_chestnut:leaf_unfolding",
    "Sycamore_maple:leaf_colouring",
    "Sycamore_maple:leaf_unfolding",
    "Willow_herb:flowering",
    "Wood_anemone:flowering",
]


seeds = {
        1: 1234,
        2: 42,
        3: 4096,
        4: 2023,
        5: 1789,
        6: 1821,
        7: 4810,
        8: 125000,
        9: 314,
        10: 10,
        11: 83083,
        12: 79073,
        13: 40491,
        14: 2692,
        15: 69490,
        16: 19883,
        17: 37784,
        18: 85096,
        19: 37964,
        20: 84116,
        21: 91344,
        22: 23167,
        23: 83994,
        24: 20989,
        25: 8493,
        26: 58112,
        27: 56551,
        28: 61215,
        29: 69487,
        30: 84939,
        31: 3193,
        32: 19621,
        33: 86889,
        34: 55429,
        35: 40502,
        36: 89292,
        37: 36035,
        38: 3468,
        39: 78507,
        40: 2360
    }


def target_list_parser(key):
    if key is None:
        return None
    if key == "PAN":
        return None
    if key == "ALL":
        target_list = []
        for s in species_table.values():
            for p in phases_table.values():
                if f"{s}:{p}" in available_species_phase:
                    target_list.append(f"{s}:{p}")
    elif ":" in key:
        s, p = key.split(":")
        target_list = [f"{species_table[s]}:{phases_table[p]}"]
    elif "+" in key:
        phases = key.split("+")
        target_list = []
        for phase in phases:
            for species_short, species_long in species_table.items():
                if f"{species_long}:{phases_table[phase]}" in available_species_phase:
                    target_list.append(f"{species_long}:{phases_table[phase]}")
    elif len(key) == 3:
        target_list = []
        for p in phases_table.values():
            candidate = f"{species_table[key]}:{p}"
            if candidate in available_species_phase:
                target_list.append(candidate)
    elif len(key) == 2:
        target_list = []
        for s in species_table.values():
            candidate = f"{s}:{phases_table[key]}"
            if candidate in available_species_phase:
                target_list.append(candidate)
    else:
        raise "target list parsing error "
    return target_list


def target_shorter(target_name):
    s, p = target_name.split(":")
    return f"{species_short[s]}:{phases_short[p]}"
