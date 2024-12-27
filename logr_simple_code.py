from math import e, log
from random import randint
# Стандартные значения из файла на гитхабе
# x1 = [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06]
# y1 = [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
# x2 = [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88]
# y2 = [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]

# Значения полученные из data_generating

x1 = [1.55, 1.64, 1.47, 0.43, 1.17, 0.86, 1.22, 0.94, 0.4, 0.82]
y1 = [0.29, 0.10, 1.42, 0.92, 1.54, 0.31, 1.89, 0.64, 1.14, 0.15]
x2 = [4.05, 6.32, 4.32, 2.12, 3.2, 1.79, 5.86]
y2 = [3.77, 4.03, 1.38, 4.58, 3.48, 4.57, 0.97]

inputs = [(x1[i], y1[i]) for i in range(len(x1))]
targets = [0 for i in range(len(x1))]
inputs += [(x2[i], y2[i]) for i in range(len(x2))]
targets += [1 for i in range(len(x2))]

weights = [randint(-100, 100) / 100 for _ in range(3)]

def weighted_z(point):
    z = [item * weights[i] for i, item in enumerate(point)]
    return sum(z) + weights[-1]

def logistic_function(z):
    return 1 / (1 + e ** (-z))

def logistic_error():
    errors = []

    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        if output == 1:
            output = 0.99999

        if output == 0:
            output = 0.00001

        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))
        errors.append(error)

    return sum(errors) / len(errors)


lr = 0.1

for epoch in range(100):
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        for j in range(len(weights) - 1):
            weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))

        weights[-1] -= lr * (output - target) * (1 / len(inputs))

    print(f"epoch: {epoch}, error: {logistic_error()}")


print(weights)


def accuracy():
    true_outputs = 0

    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        if round(output) == target:
            true_outputs += 1

    return true_outputs, len(inputs)

def test():
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]
        print(f"output: {round(output, 2)}, target: {target}")

test()
print("accuracy:", accuracy())
