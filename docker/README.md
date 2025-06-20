# SG-VLN Container

## Configuration
1. Change `ports` and `volumes` in `docker-compose.yaml`
2. run `docker compose run --name your_container_name -P vln` to start the container

## Setup dataset
1. download hm3d_val_habitat_v0.1 and objectnav_hm3d_v1 manually
2. cd to SG-VLN folder
3. make sure the dataset path in `scripts/setup_dataset.sh` is correct.
4. run `bash scripts/setup_dataset.sh`

## Run webvnc
```bash
conda activate webvnc && bash /startup.sh
```
If the script stuck just press ctrl-c to stop it and run it again. Then connect to the server using a vnc client.

## Run VLN
```bash
conda activate vln
bash run_vis_multi.sh
```