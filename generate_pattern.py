import math
import matplotlib.pyplot as plt
def generate_circle_positions(radius, center_x, center_y, num_balls):
    positions = []
    for i in range(num_balls):
        angle = 2 * math.pi * i / num_balls
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions.append((x, y))
    return positions

def generate_rectangular_positions(width, length, center_x, center_y, num_balls):
    positions = []
    for i in range(num_balls):
        x = center_x + (i % int(math.sqrt(num_balls))) * (width / math.sqrt(num_balls)) - (width / 2)
        y = center_y + (i // int(math.sqrt(num_balls))) * (length / math.sqrt(num_balls)) - (length / 2)
        positions.append((x, y))
    return positions

def generate_triangular_positions(v1, v2, v3, n):
    # Check the number of balls
    if n < 3:
        print("Sorry, n must be greater than or equal to 3.")
        return None

    # Compute the length of each edge and the perimeter
    import math
    def distance(p1, p2): # Define a function to compute the distance between two points
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    a = distance(v1, v2)  # Compute the length of the first edge
    b = distance(v2, v3)  # Compute the length of the second edge
    c = distance(v3, v1)  # Compute the length of the third edge

    perimeter = a + b + c  # 计算周长

    # Compute the number of balls on each edge
    m1 = int(round(n * a / perimeter))
    m2 = int(round(n * b / perimeter))
    m3 = n - m1 - m2

    # Initialize the list of positions
    positions = []

    # Compute the positions of the balls on each edge
    for i in range(m1):
        x = v1[0] + i * (v2[0] - v1[0]) / (m1 - 1)
        y = v1[1] + i * (v2[1] - v1[1]) / (m1 - 1)
        positions.append((x, y))

    for i in range(m2):
        x = v2[0] + i * (v3[0] - v2[0]) / (m2 - 1)
        y = v2[1] + i * (v3[1] - v2[1]) / (m2 - 1)
        positions.append((x, y))

    for i in range(m3):
        x = v3[0] + i * (v1[0] - v3[0]) / (m3 - 1)
        y = v3[1] + i * (v1[1] - v3[1]) / (m3 - 1)
        positions.append((x, y))
    return positions

def generate_line_positions (v0, v1, num_balls):
    positions = []
    for i in range(num_balls):
        x = v0[0] + (i / (num_balls - 1)) * (v1[0] - v0[0])
        y = v0[1] + (i / (num_balls - 1)) * (v1[1] - v0[1])
        positions.append((x, y))
    return positions

# Example usage:

# positions = generate_circle_positions(2, 1, 0, 20)

# positions = generate_rectangular_positions(2, 2, 0, 0, 16)

# positions = generate_triangular_positions( (0, 2), (-1, 0), (1, 0), 12)

positions = generate_line_positions([0, 0], [2, 2], 10)
print(positions)

# visualize the positions
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]

plt.scatter(x, y)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()