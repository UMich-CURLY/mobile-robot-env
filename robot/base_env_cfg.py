import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.terrains as terrain_gen

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=14,
    num_cols=14,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
        "down_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
        ),
        "up_stairs": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.25),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)
@configclass
class BaseTerminationsCfg:
    pass

@configclass
class BaseEnvCfg(LocomotionVelocityRoughEnvCfg):

    terminations: BaseTerminationsCfg = BaseTerminationsCfg()

    def load_generator(self):
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=self._physics_material,
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

    def load_usd(self, usd_path: str):
        usd_path = str(usd_path)
        self.usd_path = usd_path
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=usd_path,
            physics_material=self._physics_material,
            env_spacing=self.scene.env_spacing,
        )