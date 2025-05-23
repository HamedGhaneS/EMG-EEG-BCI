"""
Diagnostic Direction Test
Check if direction property is actually being set and applied
"""

from psychopy import visual, core, event

# Create window
win = visual.Window(
    size=[1200, 800],
    fullscr=False,
    units='height',
    color=[-0.5, -0.5, -0.5],
    allowGUI=True
)

# Create ONE test stimulus
dots = visual.DotStim(
    win,
    units='height',
    nDots=100,
    fieldShape='circle',
    fieldSize=0.4,
    dotSize=3,
    dotLife=5,
    speed=0.03,  # Even faster speed
    coherence=1.0,  # 100% coherence
    color='white',  # White dots for better visibility
    colorSpace='rgb'
)

# Test directions one by one
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

# Info display
info_text = visual.TextStim(
    win,
    text='',
    pos=(0, -0.3),
    height=0.03,
    color='yellow'
)

print("=== DIAGNOSTIC DIRECTION TEST ===")
print("Testing each direction individually...")
print("Press SPACE to cycle through directions, ESC to exit")

# Main loop
direction_timer = core.Clock()
while current_test < len(test_directions) and True:
    # Set current direction
    current_direction = test_directions[current_test]
    direction_name = direction_names[current_test]
    
    # CRITICAL: Set direction and verify it
    dots.direction = current_direction
    
    # Verify the direction was set
    actual_direction = dots.direction
    
    # Update display text
    instructions.text = f'Testing: {direction_name}\nObserve motion direction carefully'
    info_text.text = f'Set direction: {current_direction}°\nActual direction: {actual_direction}°\nPress SPACE for next test'
    
    print(f"\nTest {current_test + 1}: {direction_name}")
    print(f"  Set direction to: {current_direction}°")
    print(f"  Actual direction property: {actual_direction}°")
    print(f"  Expected motion: {direction_name}")
    
    # Draw everything
    dots.draw()
    instructions.draw()
    info_text.draw()
    win.flip()
    
    # Wait for user input
    waiting = True
    while waiting:
        keys = event.getKeys(['space', 'escape'])
        if 'space' in keys:
            current_test += 1
            waiting = False
        elif 'escape' in keys:
            waiting = False
            current_test = len(test_directions)  # Exit
        
        # Continue drawing while waiting
        dots.draw()
        instructions.draw() 
        info_text.draw()
        win.flip()
        core.wait(0.016)

win.close()
print("\n=== TEST COMPLETED ===")
print("Did you observe different motion directions for each test?")
print("If not, there may be a PsychoPy installation or graphics issue.")
