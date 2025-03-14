def probability_at_least_once(x, p=1/4096):
    return 1 - (1 - p) ** x

# Example usage:
x = 8000  # Number of tries
probability = probability_at_least_once(x)
print(f"Probability of at least one success after {x} tries: {probability * 100:.4f}%")