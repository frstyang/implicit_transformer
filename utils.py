import time
import numpy as np

class TimeContextManager:
    def __init__(self, text=None, places=2):
        self.text = text
        self.places = places
        
    def __enter__(self):
        self.time1 = time.time()
        
    def __exit__(self, type, value, trackback):
        time2 = time.time()
        time_string = format(time2 - self.time1, f".{self.places}f")
        print(f"{self.text} took {time_string}s to execute.")


def dfs(node_layers):
    names = list(node_layers.keys())
    nodes = list(node_layers.values())
    visited = {name: False for name in names}
    post_nums = {name: -1 for name in names}
    pre_nums = {name: -1 for name in names}
    clock = 0
    def post(node):
        nonlocal clock
        post_nums[node.name] = clock
        clock = clock + 1
        
    def pre(node):
        nonlocal clock
        pre_nums[node.name] = clock
        clock = clock + 1
        
    def visit(node):
        visited[node.name] = True
        pre(node)
        for child_name in node.output_names:
            if not visited[child_name]:
                visit(node_layers[child_name])
        post(node)
            
    for node in nodes:
        if not visited[node.name]:
            visit(node)
    return visited, post_nums, pre_nums

def get_back_edges(node_layers, post_nums):
    back_edges = []
    for node_name, node in node_layers.items():
        for child_name in node.output_names:
            if post_nums[child_name] > post_nums[node_name]:
                back_edges.append((node_name, child_name))
    if len(back_edges) == 0:
        print("No back edges found, is dag.")
        return []
    else:
        print(f"Not a dag! Back edges found: {back_edges}")
        return back_edges
    
def check_topo_order(node_layers, post_nums):
    prev = float(np.inf)
    for node_name, node in node_layers.items():
        if post_nums[node_name] > prev:
            print("Not in topological order!")
            return False
        prev = post_nums[node_name]
    print("In topological order!")
    return True