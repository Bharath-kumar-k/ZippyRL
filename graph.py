import json
from node import Node
from edge import Edge
from numpy import array, zeros, round_
from math import dist, atan2, pi
import matplotlib.pyplot as plt


def readFromJson(file_name):
    # Opening JSON file
    with open(file_name) as jsonFile:
        data = json.load(jsonFile)
        return data


class Graph:
    def __init__(self):
        self.nodes = [];
        self.edges = [];
        self.adjList = dict();
        self.adjMatrix = []
        self.nodeDictMap = dict()
        data = readFromJson('/home/bharath.kumar/code/ZippyRL/config/node_edge_info.json')
        self.__populateNodes(nodesMap=data['nodes'])
        self.__populateEdges(edgeMap=data['edges'])
        self.__populateAdj()

    def __populateNodes(self, nodesMap):
        # for n in nodesList:
        for idx, n in enumerate(nodesMap):
            self.nodes.append(Node(n['nodeId'], n['x'], n['y']))
            self.nodeDictMap[int(n['nodeId'])] = idx

    def __populateEdges(self, edgeMap):
        for e in edgeMap:
            self.edges.append(Edge(e['edgeId'],
                                   self.nodes[self.nodeDictMap[int(e['fromNode'])]],
                                   self.nodes[self.nodeDictMap[int(e['toNode'])]]))

    def __populateAdj(self):
        self.adj_matrix = zeros((len(self.nodes), len(self.nodes)))
        for n in self.nodes:
            self.adjList[n.nodeId()] = []

        for e in self.edges:
            fromNode = e.fromNode()
            toNode = e.toNode()

            pos = e.edgePos()

            edgeDist = dist(pos[0], pos[1])
            edgeAng = round_(atan2(pos[1, 1] - pos[0, 1], pos[1, 0] - pos[0, 0]) / pi, 1)

            i = fromNode.nodeId()
            j = toNode.nodeId()

            self.adj_matrix[i, j] = 1

            # update edge cost
            e.edgeCost(edgeDist)

            # update edge dist
            e.edgeDist(edgeDist)

            # update edge angle
            e.edgeAng(edgeAng)

            # updating adj list
            self.adjList[i].append(j)

            # update inflow edge for each node
            toNode.appendInflowEdge(e.edgeId())
            # update outflow edge for each node
            fromNode.appendOutflowEdge(e.edgeId())

            # update inflow node for each node
            toNode.appendIncomingNodes(fromNode.nodeId())
            # update outflow node for each node
            fromNode.appendOutgoingNodes(toNode.nodeId())

    def plot_graph(self, show_all_edge=False, show_node_name=False, alpha=0.5):

        # fig, ax = plt.subplots(figsize=(12, 12))
        fig, (axAnim, axDisToGo, axVec, axAcc, axAct) = plt.subplots(nrows=5, figsize=(13, 13))
        axAnim.set_aspect('equal')
        nodePos = array([n.nodePos() for n in self.nodes])
        axAnim.scatter(nodePos[:, 0], nodePos[:, 1], alpha=alpha)

        if show_node_name:
            for n in self.nodes:
                name, x, y = n.nodePos(return_with_node_id=True)
                axAnim.annotate(name, (x, y + 0.05))

        if show_all_edge:
            for e in self.edges:
                edgePos = e.edgePos()
                axAnim.arrow(edgePos[0, 0], edgePos[0, 1], edgePos[1, 0] - edgePos[0, 0],
                         edgePos[1, 1] - edgePos[0, 1],
                         head_width=0.1)

        time_text = axAnim.text(1.005, 0.60, '', transform=axAnim.transAxes)
        acc_text = axAnim.text(1.005, 0.50, '', transform=axAnim.transAxes)
        vel_text = axAnim.text(1.005, 0.40, '', transform=axAnim.transAxes)
        disToGo_text = axAnim.text(1.005, 0.30, '', transform=axAnim.transAxes)
        return fig, axAnim, axDisToGo, axVec, axAcc, axAct, time_text, acc_text, vel_text, disToGo_text

# g = Graph()
# g.plot_graph(show_all_edge=True, show_node_name=True)
# plt.show()
