import json

def load_pokemon_data(filename="pokemon_data.json"):
    """Reads the Pokémon data from a JSON file and returns it as a list of dictionaries."""
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' contains invalid JSON.")
        return []

def has_type(pokemon, type_name):
    """Checks if a given Pokémon has a specified type."""
    return type_name.lower() in [t.lower() for t in pokemon["types"]]

# Example usage
pokemon_data = load_pokemon_data()

for pokemon in pokemon_data:
    if has_type(pokemon, "ice"):
        print(f"{pokemon['name']} is a Grass-type Pokémon!")