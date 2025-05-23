"""
Alternative Motion Test - Circle moves and disappears when it hits an invisible border
"""

from psychopy import visual, core, event
import math

# === Settings ===
boundary_shape = "circle"  # "circle" or "square"
boundary_size = 0.2        # Radius (for circle) or half-width (for square)
circle_radius = 0.01
motion_speed = 0.005

# Create window
win = visual.Window(
    size=[1200, 800],
    fullscr=False,
    units='height',
    color=[-0.5, -0.5, -0.5],
    allowGUI=True
)

# Moving circle
moving_circle = visual.Circle(
    win,
    radius=circle_radius,
    fillColor='white',
    lineColor='white'
)

# Directions
test_directions = [0, 90, 180, 270]
direction_names = ['RIGHT (0°)', 'UP (90°)', 'LEFT (180°)', 'DOWN (270°)']
current_test = 0

# Instructions
instructions = visual.TextStim(
    win,
    text='',
    pos=(0, 0.3),
    height=0.04,
    color='white'
)

info_text = visual.TextStim(
    win,
    text='',
    pos=(0, -0.3),
    height=0.03,
    color='yellow'
)

print("=== INVISIBLE BOUNDARY MOTION TEST ===")
print("Circle moves and disappears at the edge of an invisible boundary")
print("Press SPACE to cycle through directions, ESC to exit")

# === Main Loop ===
while current_test < len(test_directions):
    direction_degrees = test_directions[current_test]
    direction_radians = math.radians(direction_degrees)
    direction_name = direction_names[current_test]

    dx = motion_speed * math.cos(direction_radians)
    dy = motion_speed * math.sin(direction_radians)

    x, y = 0, 0  # Start from center
    visible = True  # Circle starts visible

    print(f"\nTest {current_test + 1}: {direction_name}")

    while True:
        keys = event.getKeys(['space', 'escape'])
        if 'space' in keys:
            current_test += 1
            break
        elif 'escape' in keys:
            current_test = len(test_directions)
            break

        if visible:
            # Update position
            x += dx
            y += dy

            # Check boundary
            if boundary_shape == "square":
                if abs(x) + circle_radius > boundary_size or abs(y) + circle_radius > boundary_size:
                    visible = False
            else:  # circle boundary
                if math.sqrt(x**2 + y**2) + circle_radius > boundary_size:
                    visible = False

        # Draw
        if visible:
            moving_circle.pos = (x, y)
            moving_circle.draw()

        instructions.text = f'Testing: {direction_name}\nWatch the circle move and vanish'
        info_text.text = f'Direction: {direction_degrees}°\nPress SPACE for next, ESC to exit'
        instructions.draw()
        info_text.draw()
        win.flip()

        core.wait(0.016)

win.close()
print("\n=== TEST COMPLETED ===")
print("Did the circle disappear at the correct location?")
