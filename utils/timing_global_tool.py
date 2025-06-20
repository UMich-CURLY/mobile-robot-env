import numpy as np

# These variables are used as global variables to store the time taken by each module
ALL_TIME = []

# this is used for compuating average time for each episode
PERCEPTION_TIME_EPISODE = []
MAPPING_TIME_EPISODE = []
SCENE_GRAPH_TIME_EPISODE = []
PLANNING_TIME_EPISODE = []
LLM_TIME_EPISODE = []


def init_episode_entry():
    '''
    Clear the time taken by each module for each episode
    '''
    PERCEPTION_TIME_EPISODE.clear()
    MAPPING_TIME_EPISODE.clear()
    SCENE_GRAPH_TIME_EPISODE.clear()
    PLANNING_TIME_EPISODE.clear()
    LLM_TIME_EPISODE.clear()

def add_new_time_entry(total_step):
    '''
    Add the time taken by each module for the current episode
    '''
    ALL_TIME.append({})
    
    ALL_TIME[-1]["perception"] = np.sum(np.array(PERCEPTION_TIME_EPISODE))
    ALL_TIME[-1]["mapping"] = np.sum(np.array(MAPPING_TIME_EPISODE))
    ALL_TIME[-1]["scene_graph"] = np.sum(np.array(SCENE_GRAPH_TIME_EPISODE))
    ALL_TIME[-1]["planning"] = np.sum(np.array(PLANNING_TIME_EPISODE))
    ALL_TIME[-1]["llm"] = np.sum(np.array(LLM_TIME_EPISODE))
    ALL_TIME[-1]["total_nav_step"] = total_step
    

def add_new_episode_entry():
    '''
    For each step in the episode, add the time taken by each module
    '''
    PERCEPTION_TIME_EPISODE.append(0)
    MAPPING_TIME_EPISODE.append(0)
    SCENE_GRAPH_TIME_EPISODE.append(0)
    PLANNING_TIME_EPISODE.append(0)
    LLM_TIME_EPISODE.append(0)
