import csv

def load_csv(filename):
    """Load training examples from a CSV file."""
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def candidate_elimination(training_data):
    num_attributes = len(training_data[0]) - 1
    
    # Initialize specific hypothesis (S) with the first positive example
    S = None
    for example in training_data:
        if example[-1] == "Yes":
            S = example[:-1]
            break
    
    if S is None:
        print("No positive examples found. Version space is empty.")
        return None, None
    
    # Initialize general hypothesis (G) as maximally general
    G = [['?' for _ in range(num_attributes)]]
    
    print("Initial Specific Boundary (S):", S)
    print("Initial General Boundary (G):", G)
    print("-" * 50)
    
    # Process each example step by step
    for idx, example in enumerate(training_data, start=1):
        attributes, label = example[:-1], example[-1]
        print(f"Example {idx}: {attributes} -> {label}")
        
        if label == "Yes":  # Positive example
            # Update S
            for i in range(num_attributes):
                if S[i] != attributes[i]:
                    S[i] = '?'
            
            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[i] == '?' or g[i] == attributes[i] for i in range(num_attributes))]
        
        else:  # Negative example
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == '?':
                        if S[i] != attributes[i]:
                            new_hypothesis = g.copy()
                            new_hypothesis[i] = S[i]
                            new_G.append(new_hypothesis)
            G = new_G
        
        print("Updated Specific Boundary (S):", S)
        print("Updated General Boundary (G):", G)
        print("-" * 50)
    
    return S, G

# Load data from data.csv
data = load_csv("data.csv")

# Run Candidate-Elimination
S, G = candidate_elimination(data)

print("Final Specific Boundary (S):", S)
print("Final General Boundary (G):", G)