# Overview animations

- `city.gif`: 800 × 500, 144 frames, 12 seconds, infinite loop. Several city routes → teacher selects the green route → student follows it → student collects the diamond.
- `desert.gif`: 800 × 500, 144 frames, 12 seconds, infinite loop. Three candidate paths → teacher selects the tangled middle route → student follows multiple self-crossing loops and doubles back, without reaching the shelter from the original overview.
- `city-poster.png`, `desert-poster.png`: still alternatives for reduced-motion preferences.
- `shelter.png`: isolated cottage and adjacent trees derived from the original desert target with the built-in imagegen tool.
- `teacher.png`, `student.png`: transparent character assets, extracted from `characters.png`.
- `characters.png`: character sheet derived from the original `../figures/overview.png` using the built-in imagegen tool. The source overview image is preserved.

Scenes, paths, character motion, timing, captions and GIF encoding are built by `scripts/render-overview-animations.cjs` using resvg and gifenc. Run the commands at the top of that file to rebuild. GIFs work without JavaScript; the page script adds still-image controls.

## Imagegen prompt

### Shelter update

Use case: background-extraction. Edit target: the supplied original overview illustration. Extract ONLY the small shelter/house at the distant target in the RIGHT-HAND DESERT PANEL (under the TARGET sign, around x=1365,y=110 in the reference). Preserve that exact cottage design: warm cream walls, ochre brown pitched gable roof, simple brown outlines, tiny dark windows, front doorway, its original hand-drawn watercolor storybook look, with the two small olive-green trees beside the house. Output one isolated shelter with these adjacent trees, centered, complete and uncut, on a truly transparent background. No sign, no TARGET text, no desert landscape, no teacher, no student, no diamond, no mountains, no labels. Do not redesign the house. Crop closely with a little transparent padding; landscape composition.

### Characters

Use case: background-extraction / identity-preserve. Extract the two character designs from the LEFT CITY half of the reference as a clean transparent animation sprite sheet. Exactly two isolated full body characters: LEFT HALF the adult teacher, brown messy hair, spectacles, olive brown long coat, brown boots, holding the city map in his left hand and pointing to the right with his right hand. RIGHT HALF the young student, brown messy hair, blue hoodie, navy backpack, dark teal trousers and brown shoes, facing right in a walking pose. Preserve their original charming hand-drawn storybook linework, watercolor shading, face, clothes and colors. No city, no background, no ground, no pedestal, no diamond, no speech bubbles or words (map may have simple grid only). Actual transparent alpha background. Wide landscape 3:2 composition, teacher wholly inside left half and student wholly inside right half with ample empty transparent margin; no overlap. Both characters' feet at same baseline at 90% height, teacher taller than student as in reference. The two halves will be separately clipped and animated in a webpage.
