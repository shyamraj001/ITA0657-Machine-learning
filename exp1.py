def find_s(training_data):
    hypothesis = ['0'] * (len(training_data[0]) - 1)

    print("Initial Hypothesis:", hypothesis)

    for i, example in enumerate(training_data):
        if example[-1].lower() == "yes":  # Consider only positive examples
            print(f"\nProcessing Positive Example {i+1}: {example[:-1]}")

            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = example[j]
                elif hypothesis[j] != example[j]:
                    hypothesis[j] = '?'

            print("Updated Hypothesis:", hypothesis)

    return hypothesis

training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Run FIND-S
final_hypothesis = find_s(training_data)

print("\nFinal Hypothesis:", final_hypothesis)