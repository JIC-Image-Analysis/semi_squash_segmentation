FCA alleles
===========

Image conversion
----------------

~/packages/bftools-5.5.3/bfconvert ~/data_repo/smFISH_semi_squashes/data/ColFRI-semisq_PP2A_FLC_02.czi data_intermediate/ColFRI-semisq_PP2A_FLC_02/converted_S%s_C%c_Z%z.png

Manual marker image generation
------------------------------

1. Load image
2. Select multiple point tool
3. Click points
4. Analyse->Measure
5. File->Save (as CSV)


Pipelines
---------


Step 1: generate_image_for_labelling
Step 2: manually label and produce CSV
Step 3: convert_nuclei_centroids_to_voronoi
