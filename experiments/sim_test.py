from habitat import Env
from habitat.config.default import get_config
from habitat.config import read_write
import yaml
from omegaconf import OmegaConf, DictConfig
def print_scene_recur(scene, limit_output=10):
    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None
                
# config = get_config(config_paths='/workspace_sdc/tiamat_ws/SG-VLN/configs/objectnav_hm3d.yaml')
# config = get_config('configs/objectnav_hssd-hab_rgbd_semantic.yaml')
config = get_config('configs/objectnav_hm3d_rgbd_semantic.yaml')

with read_write(config):
    config.habitat.dataset.split = "val"
    config.habitat.environment.iterator_options.shuffle = False
    
data = OmegaConf.to_yaml(config)
print(type(data))
yaml_data = yaml.safe_load(data)
print(type(yaml_data))
with open('semantic_hm3d_config.yaml', 'w') as outfile:
    yaml.dump(yaml_data, outfile)
    
# raise ValueError("Config file is not correct")

# config.defrost()
# config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# config.freeze()
env = Env(config)



print("--------------- Printing Semnatic Annotations ---------------")
scene = env.sim.semantic_annotations()
print(scene.objects)
# print_scene_recur(scene, limit_output=15)

env.close()