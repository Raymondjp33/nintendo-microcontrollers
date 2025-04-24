# Full Pokémon type chart (lowercase)
type_chart = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"grass": 2.0, "ice": 2.0, "bug": 2.0, "steel": 2.0, "fire": 0.5, "water": 0.5, "rock": 0.5, "dragon": 0.5},
    "water": {"fire": 2.0, "ground": 2.0, "rock": 2.0, "water": 0.5, "grass": 0.5, "dragon": 0.5},
    "electric": {"water": 2.0, "flying": 2.0, "electric": 0.5, "grass": 0.5, "dragon": 0.5, "ground": 0.0},
    "grass": {"water": 2.0, "ground": 2.0, "rock": 2.0, "fire": 0.5, "grass": 0.5, "poison": 0.5, "flying": 0.5, "bug": 0.5, "dragon": 0.5, "steel": 0.5},
    "ice": {"grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "fire": 0.5, "water": 0.5, "ice": 0.5, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "rock": 2.0, "dark": 2.0, "steel": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "fairy": 0.5, "ghost": 0.0},
    "poison": {"grass": 2.0, "fairy": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0},
    "ground": {"fire": 2.0, "electric": 2.0, "poison": 2.0, "rock": 2.0, "steel": 2.0, "grass": 0.5, "bug": 0.5, "flying": 0.0},
    "flying": {"grass": 2.0, "fighting": 2.0, "bug": 2.0, "electric": 0.5, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "steel": 0.5, "dark": 0.0},
    "bug": {"grass": 2.0, "psychic": 2.0, "dark": 2.0, "fire": 0.5, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "ghost": 0.5, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0, "fighting": 0.5, "ground": 0.5, "steel": 0.5},
    "ghost": {"psychic": 2.0, "ghost": 2.0, "dark": 0.5, "normal": 0.0},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"psychic": 2.0, "ghost": 2.0, "fighting": 0.5, "dark": 0.5, "fairy": 0.5},
    "steel": {"ice": 2.0, "rock": 2.0, "fairy": 2.0, "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
    "fairy": {"fighting": 2.0, "dragon": 2.0, "dark": 2.0, "fire": 0.5, "poison": 0.5, "steel": 0.5}
}

all_types = list(type_chart.keys())

def get_defensive_multiplier(pokemon_type, attacking_types, type_chart):
    if not attacking_types:
        return 0.0  # no threats
    return sum(type_chart.get(atk, {}).get(pokemon_type, 1.0) for atk in attacking_types)

def get_offensive_multiplier(pokemon_type, defending_types, type_chart):
    if not defending_types:
        return 1.0
    multiplier = 1.0
    for target in defending_types:
        multiplier *= type_chart.get(pokemon_type, {}).get(target, 1.0)
    return multiplier

def rank_types(defense_types, offense_types, type_chart):
    defense_types = [t.lower() for t in defense_types]
    offense_types = [t.lower() for t in offense_types]
    type_scores = []
    for t in all_types:
        defense_score = get_defensive_multiplier(t, defense_types, type_chart)
        offense_score = get_offensive_multiplier(t, offense_types, type_chart)
        type_scores.append((t, defense_score, offense_score))
    
    type_scores.sort(key=lambda x: (x[1], -x[2]))  # prioritize low defense score, then high offense
    return type_scores

# Example usage
if __name__ == "__main__":
    defense = ["ice", "flying"]
    offense = ["ice", "flying"]

    ranked = rank_types(defense, offense, type_chart)
    for t, d_score, o_score in ranked:
        # print(f"{t}: Defense Score = {d_score:.2f}, Offense Score = {o_score:.2f}")
        print(f'"{t}",')