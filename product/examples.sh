source activate dynaslum2


#python analysis.py  data/section_1.tif data/slums_approved.shp features/section_1__BD5-3-2_BK24_SC44_TRhog/merged.tif --block_size 24
#python analysis.py  data/section_1_fixed.tif data/slums_approved.shp features/section_1_fixed__BD5-3-2_BK24_SC24_TRlsr/merged.tiff --block_size 24
#python analysis.py  data/section_1_fixed.tif data/slums_approved.shp features/section_1_fixed__BD5-3-2_BK24_SC44_TRhog/merged.tiff --block_size 24
#python analysis.py  data/section_1_fixed.tif data/slums_approved.shp features/section_1_fixed__BD5-3-2_BK40_SC200_TRhog/merged.tiff --block_size 40
#python analysis.py  data/section_1_fixed.tif data/slums_approved.shp features/section_1_fixed__BD5-3-2_BK20_SC50_TRhog/merged.tiff --block_size 20
#python analysis.py  data/section_1_fixed.tif data/slums_approved.shp features/section_1_fixed__BD5-3-2_BK20_SC20_TRhog-lsr/merged.tiff --block_size 20
python analysis.py  data/section_1_fixed.tif data/slums_approved.shp features/section_1_fixed__BD5-3-2_BK20_SC50_TRhog-lsr/merged.tiff --block_size 20

source deactivate