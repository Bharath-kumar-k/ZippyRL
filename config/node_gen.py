import numpy as np
import json


def get_highway_data(highway_file_name):
    with open(highway_file_name, "r") as file:
        all_line = file.readlines()
    file.close()
    return np.array([int(l.strip().split()[1]) for l in all_line])


def get_env_data(env_file_name):
    with open(env_file_name, "r") as file:
        all_line = file.readlines()
    file.close()
    return np.array([[int(x) for x in l.strip().split()] for l in all_line[10:]]), float(all_line[4].split()[1])


def get_graph_info(env, d_x_y, hw_x, hw_y, write_json=False):
    env_shape = env.shape
    env_x = np.array([[(j) * d_x_y for j in range(env_shape[1])] for i in range(env_shape[0])])
    env_y = np.array([[(i) * d_x_y for j in range(env_shape[1])] for i in range(env_shape[0])])

    # 3rd dim [horizontal, vertical]
    # horizontal 1: connect to right node, -1 connect to left node
    # vertical 1: connect to top node, -1 connect to bottom node
    env_edge_info = np.zeros((env_shape[0] + 2, env_shape[1] + 2, 2), dtype=int)
    env_edge_info[1:env_shape[0] + 1, 1:env_shape[1] + 1, 0] = hw_x.reshape((-1, 1)) * (env ^ 1)
    env_edge_info[1:env_shape[0] + 1, 1:env_shape[1] + 1, 1] = hw_y.reshape((1, -1)) * (env ^ 1)

    graph_info = {"nodes": [],
                  "edges": []}

    node_itr = 0
    node_id_matrix = (-1) * np.ones_like(env)
    for i in np.arange(env_shape[0]):
        for j in np.arange(env_shape[1]):
            if env[i, j] == 0:
                # Adding node info
                graph_info["nodes"].append(dict(nodeId=node_itr, x=env_x[i, j], y=env_y[i, j]))
                node_id_matrix[i, j] = node_itr
                node_itr += 1
    edge_itr = 0
    for i in np.arange(1, env_shape[0] + 1):
        for j in np.arange(1, env_shape[1] + 1):
            if env[i - 1, j - 1] == 0:
                # Adding edge info

                # Horizontal_edge
                if env_edge_info[i, j, 0] == 1 and env_edge_info[i, j+1, 0] == 1:
                    from_node = node_id_matrix[i - 1, j - 1]
                    to_node = node_id_matrix[i-1, j]

                    graph_info["edges"].append(dict(edgeId=edge_itr,
                                                    fromNode=int(from_node),
                                                    toNode=int(to_node)))
                    edge_itr += 1
                if env_edge_info[i, j, 0] == -1 and env_edge_info[i, j - 1, 0] == -1:
                    from_node = node_id_matrix[i - 1, j - 1]
                    to_node = node_id_matrix[i - 1, j - 2]

                    graph_info["edges"].append(dict(edgeId=edge_itr,
                                                    fromNode=int(from_node),
                                                    toNode=int(to_node)))
                    edge_itr += 1

                # Vertical_edge
                if env_edge_info[i, j, 1] == 1 and env_edge_info[i + 1, j, 1] == 1:
                    from_node = node_id_matrix[i - 1, j - 1]
                    to_node = node_id_matrix[i, j - 1]

                    graph_info["edges"].append(dict(edgeId=edge_itr,
                                                    fromNode=int(from_node),
                                                    toNode=int(to_node)))
                    edge_itr += 1
                if env_edge_info[i, j, 1] == -1 and env_edge_info[i - 1, j, 1] == -1:
                    from_node = node_id_matrix[i - 1, j - 1]
                    to_node = node_id_matrix[i - 2, j - 1]

                    graph_info["edges"].append(dict(edgeId=edge_itr,
                                                    fromNode=int(from_node),
                                                    toNode=int(to_node)))
                    edge_itr += 1

    if write_json:
        #json_node_edge_object = json.dumps(graph_info, indent=4)
        with open("/home/bharath.kumar/code/ZippyRL/config/node_edge_info.json", "w") as outfile:
            json.dump(graph_info, outfile)

    return graph_info


# get x highway information
highway_x = get_highway_data('/home/bharath.kumar/code/ZippyRL/config/highway_x.txt')

# get y highway information
highway_y = get_highway_data('/home/bharath.kumar/code/ZippyRL/config/highway_y.txt')

# get environment data
environment_data, dxy = get_env_data('/home/bharath.kumar/code/ZippyRL/config/env_sorter.cfg')

get_graph_info(environment_data, dxy, highway_x, highway_y, write_json=True)