# Run the visualization with multi-GPU support in the Singularity container
# singularity exec --nv  --fakeroot \
#     -B /home/junzhewu/Projects/VLN-Game:/VLN:rw \
#     vln_sg_nav.sif bash /VLN/run_vis_multi.sh
# singularity exec --nv \
#     -B /home/junzhewu/Projects/VLN-Game:/VLN:rw \
#     vln_sg_nav.sif bash
singularity exec --nv \
    -B /home/junzhewu/Projects/VLN-Game:/VLN:rw \
    vln_sg_nav.sif bash