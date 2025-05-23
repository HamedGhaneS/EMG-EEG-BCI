"""
Simple Direction Test - Two circles side by side
Using EXACT same DotStim setup as Phase 0 script
Left circle: LEFT motion (direction=180)
Right circle: RIGHT motion (direction=0)
"""

from psychopy import visual, core, event

# Create window - SAME as Phase 0
win = visual.Window(
    size=[1200, 800],
    fullscr=False,
    units='height',
    color=[-0.5, -0.5, -0.5],  # Gray background
    allowGUI=True
)

# EXACT same settings as Phase 0 main script
settings = {
    'dot_size': 3,
    'dot_lifetime': 5,
    'n_dots': 100,
    'dot_speed': 0.01,
}
config = {
    'aperture_size': 0.4,
    'aperture_shape': 'square',  # CHANGE THIS: 'circle' or 'square'
    'high_coherence': 1,
}

# LEFT motion stimulus - EXACT same setup as Phase 0
left_dots = visual.DotStim(
    win,
    units='height',
    nDots=settings['n_dots'],
    fieldShape=config['aperture_shape'],  # Use configured shape
    fieldSize=config['aperture_size'],
    dotSize=settings['dot_size'],
    dotLife=settings['dot_lifetime'],
    speed=settings['dot_speed'],
    coherence=config['high_coherence'],
    color='black',
    colorSpace='rgb'
)
# Set direction and position after creation - FIXED MAPPING
left_dots.direction = 0    # FIXED: Use 0 to get LEFTWARD motion
left_dots.pos = (-0.3, 0)  # Left side

# RIGHT motion stimulus - EXACT same setup as Phase 0
right_dots = visual.DotStim(
    win,
    units='height',
    nDots=settings['n_dots'],
    fieldShape=config['aperture_shape'],  # Use configured shape
    fieldSize=config['aperture_size'],
    dotSize=settings['dot_size'],
    dotLife=settings['dot_lifetime'],
    speed=settings['dot_speed'],
    coherence=config['high_coherence'],
    color='black',
    colorSpace='rgb'
)
# Set direction and position after creation - FIXED MAPPING
right_dots.direction = 180  # FIXED: Use 180 to get RIGHTWARD motion
right_dots.pos = (0.3, 0)   # Right side

# Labels
left_label = visual.TextStim(
    win,
    text=f'LEFT\n(direction=0)\nShape: {config["aperture_shape"]}',
    pos=(-0.3, -0.25),
    height=0.04,
    color='white'
)

right_label = visual.TextStim(
    win,
    text=f'RIGHT\n(direction=180)\nShape: {config["aperture_shape"]}',
    pos=(0.3, -0.25),
    height=0.04,
    color='white'
)

# Instructions
instructions = visual.TextStim(
    win,
    text='Direction Test: LEFT circle should move LEFT, RIGHT circle should move RIGHT\nPress SPACE to exit',
    pos=(0, 0.35),
    height=0.03,
    color='white'
)

print("FIXED Direction Test Running...")
print("LEFT circle: direction=0 (should move LEFT)")
print("RIGHT circle: direction=180 (should move RIGHT)")
print("Press SPACE to exit")

# Main loop
while True:
    # Draw everything
    left_dots.draw()
    right_dots.draw()
    left_label.draw()
    right_label.draw()
    instructions.draw()
    
    win.flip()
    
    # Check for exit
    keys = event.getKeys(['space', 'escape'])
    if keys:
        break
    
    core.wait(0.016)  # ~60 FPS

win.close()
print("Test completed")
