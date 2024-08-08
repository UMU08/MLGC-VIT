import numpy as np
with open('record/SEED_UPDATE_LOG1.txt') as input_file, open('record/output.txt', 'w') as output_file:
    data = {}
    for line in input_file:
        parts = line.strip().split()
        if len(parts) >= 2:
            people = parts[1]
            value = float(parts[-1])
            if people in data:
                if value > data[people]:
                    data[people] = value
            else:
                data[people] = value
    print(data)
    group = 0
    current_group = []
    ans = []
    sum = 0
    Sum = 0
    for people, value in data.items():
        current_group.append(value)
        if len(current_group) == 3:
            current_group.sort(reverse=True)
            sum += (current_group[0] + current_group[1])
            ans.append(current_group[0])
            ans.append(current_group[1])
            Sum += (current_group[0] + current_group[1] + current_group[2])
            current_group = []
    print(Sum, Sum / 45)
    print(sum, sum / 28)
    print(len(ans))
    print("std:",np.std(ans))

