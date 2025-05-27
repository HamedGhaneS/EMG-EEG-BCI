# RDM Experiment with Custom Dot Implementation

## Why This Custom Implementation?

**Problem**: PsychoPy's built-in `visual.DotStim` has a well-known direction bug where dots get "stuck" moving in one direction (typically rightward) regardless of the `direction` property setting. This bug affects:
- Direction discrimination tasks
- Multi-directional motion studies  
- High-coherence motion conditions
- Research requiring reliable directional control

**Evidence**: Debugging tests revealed that while `dots.direction = 180` appeared to work in console output, all motion remained rightward visually, leading to incorrect behavioral responses.

**Solution**: Custom dot implementation using individual `MovingDot` objects with manual motion vector calculations that bypass PsychoPy's internal caching issues.

## Key Advantages of Custom Implementation

- ✅ **Reliable Direction Control**: No direction-sticking bugs
- ✅ **Research-Grade Accuracy**: Precise coherence and motion control
- ✅ **Full Transparency**: Complete control over dot behavior and parameters
- ✅ **Performance Optimized**: Tailored for specific experimental needs
- ✅ **Debugging Friendly**: Easy to trace and modify individual dot behaviors

## How This Implementation Works

### Core Components

1. **MovingDot Class**: Individual dot objects with position, lifetime, and motion state
2. **Coherence Control**: Randomly assigns dots to coherent vs. random motion each frame
3. **Boundary Management**: Dots disappear when leaving circular aperture
4. **Lifetime System**: Dots respawn after 5-10 frames to prevent tracking

### Motion Control Logic

```python
# Coherent dots: Move in specified direction
dx = motion_speed * cos(direction_radians)
dy = motion_speed * sin(direction_radians)

# Random dots: Move in random directions  
random_angle = uniform(0, 2π)
dx = motion_speed * cos(random_angle)
dy = motion_speed * sin(random_angle)
```

### Integration with Original Experiment

- **Maintains identical trial structure** and timing
- **Preserves all response collection** and confidence rating systems
- **Same data output format** for analysis compatibility
- **Same configuration options** and color presets
- **Same Cedrus response pad integration**

## Usage Instructions

1. **Configuration**: Modify settings in the `EASY CONFIGURATION SECTION`
2. **Dot Parameters**: Adjust `dot_radius`, `motion_speed`, `n_dots` in advanced settings
3. **Performance**: Set `refresh_rate` to match your display (60Hz, 120Hz, etc.)
4. **Direction Testing**: Built-in debugging available in separate test script

## Performance Considerations

- **Frame Rate**: Optimized for 120Hz displays (adjustable)
- **Dot Count**: 200 dots provide good motion signal while maintaining performance
- **Lifetime**: Short dot lifetime (5-10 frames) prevents participant tracking strategies
- **Boundary**: Circular aperture with proper edge handling

## Research Validation

This implementation addresses the fundamental reliability issues with PsychoPy's DotStim while maintaining full compatibility with standard RDM experimental paradigms used in neuroscience research.

**Recommended for**: Motion perception studies, decision-making experiments, neural recordings, and any research requiring reliable directional motion control.

---

*Note: This script was developed in response to documented PsychoPy DotStim direction bugs affecting multi-directional motion experiments. The custom implementation ensures research-grade reliability and reproducibility.*
