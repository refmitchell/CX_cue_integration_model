## Multimodal cue integration in the insect central complex

This repository accompanies the manuscript "A model of cue integration as vector summation in the insect brain". All relevant data and sofware are included in this repository.

## Software
All python code used for simulations and plotting can by found in the `python` subdirectory. 
The Ring Model itself is laid out in `extended_ring_model.py` and our simulation routine is
detailed in `simulation_utilities.py'.

## Data
Within the python subdirectory there are four 'data' directories:

1. `behavioural_data` includes our new behavioural data.
2. `data_out` is the standard simulation output directory.
3. `dacke2019_data` contains data from Dacke et al. (2019), Multimodal cue integration in the dung beetle compass.
4. `shav2022_data` contains data from Shaverdian, Dirlik, Mitchell, et al. (2022), Weighted cue integration for straight-line orientation

We would like to emphasise, (3) and (4) contain data from previous publications
(included with author permission), this is not new data. They are referenced
here and in the manuscript.

## Legacy items
Any item marked `[Legacy]` was not included in any way in the final manuscript. These are usually
things which were tried or considered at one time or another and can be useful to include
for completeness' sake. If `[Legacy]` appears in the module header comment, then nothing in
that module was used in the manuscript (supplementary information).

## Figure production
All data figures are output in the `python/plots`
subdirectory. In some cases, modifications were made in Inkscape to
combine multiple figures (or fix fonts/labelling). The SVG files can
be found in the `figs` subdirectory.

## Contact
If you have questions about any aspect of this repository, please
contact r.mitchell@ed.ac.uk.
