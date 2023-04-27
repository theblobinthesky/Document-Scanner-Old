TODO: Must have performance improvements and general engine changes aswell as visual changes.
* Profiling?
* More composable graphics, especially respecting rounded corners
* Gradient rendering, SDF animations
* Asset packing, using mmap avoiding memcpy
* Blur transitions  
* Port platform layer to linux
See https://www.shadertoy.com/view/WsVGWV and https://github.com/mattdesl/lwjgl-basics/wiki/OpenGL-ES-Blurs#LerpBlur.

TODO: Asset packing
* SDF generator from binary masks  
See https://tkmikyon.medium.com/computing-the-signed-distance-field-a1fa9ba2fc7d for reference.
* Asset pack with SDFs.

TODO: UI
* Checkboxes using SDF animation,
* Three stripes to open selection panel
* Fix button click animations
* Discard button should actually do something
* Export menu should load icons of destinations and names
* Export menu should open subdialogs when necessary