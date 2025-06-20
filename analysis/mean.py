import os
from argparse import ArgumentParser
from load import load_all_info
import numpy as np
import pandas
from IPython.display import display
# from tabulate import tabulate
parser = ArgumentParser()
parser.add_argument('dir', type=str,  metavar='N', nargs='+',
                   )
parser.add_argument('--all',action='store_true')
args = parser.parse_args()

# directories = [os.path.join(args.dir, d) for d in os.listdir(args.dir)]
directories = []
exp_names = []

inputlist = args.dir

if(args.all):
    inputlist = [os.path.join(args.dir[0], d) for d in os.listdir(args.dir[0])]
for directory in inputlist:
    goal = directory
    if os.path.basename(directory) != "logs":
        if os.path.basename(os.path.dirname(directory)) != "objectnav-dino":
            goal= os.path.join(goal,'objectnav-dino')
        goal =  os.path.join(goal,"logs/")
    directories.append(goal)
    exp_names.append(os.path.basename(os.path.dirname(os.path.dirname(goal[:-1]))))

info_list = []

for directory in directories:
    info_list.append(load_all_info(directory))

spl = []
success =[]

def calc_spl(results):
    spl_list = []
    num_success = 0
    for result in results:
        # if result['spl'] > 0:
        if result['habitat_success'] == 1:
            num_success += 1
        if np.isnan(result['spl']):
            spl_list.append(0)
        else:
            spl_list.append(result['spl'])


    spl.append(np.mean(spl_list))
    # print("spl:", np.mean(spl_list))
    # print("success rate:", num_success / len(results))
    if(len(results)>0):
        success.append(num_success / len(results))
    else:
        success.append("invalid")

def find_common_element_indices(lists):
    if not lists:
        return {}
    common_elements = set(lists[0]).intersection(*lists[1:])
    print(common_elements)
    indices = [[] for i in range(len(lists))]
    for element in common_elements:
        for index,lst in enumerate(lists):
            indices[index].append(lst.index(element))
    return indices
if not args.all:
    episodes = []
    for info in info_list:
        episodes.append([])
        for result in info:
            episodes[-1].append(result['episode'])
    indices = find_common_element_indices(episodes)

    print(indices)
    for i,info in enumerate(info_list):
        info_list[i] = [info[j] for j in indices[i]]


for info in info_list:
    calc_spl(info)

dict = {'name': exp_names, 'success': success, 'spl':spl} 
df = pandas.DataFrame(dict)
display(df)

# print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
