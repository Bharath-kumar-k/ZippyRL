from numpy import array


class Node:
    def __init__(self, nodeId=None, x=None, y=None):
        self.__nodeId = nodeId
        self.__x = x
        self.__y = y
        self.__incomingNodes = []
        self.__outgoingNodes = []
        self.__inflowEdge = []
        self.__outflowEdge = []

    def __eq__(self, other):
        return True if self.__nodeId == other.__nodeId else False

    def nodePos(self, return_with_node_id=False):
        return (self.__nodeId, self.__x, self.__y) if return_with_node_id else (self.__x, self.__y)

    def nodeId(self):
        return self.__nodeId

    def incomingNodes(self):
        return array(self.__incomingNodes)

    def outgoingNodes(self):
        return array(self.__outgoingNodes)

    def inflowEdge(self):
        return array(self.__inflowEdge)

    def outflowEdge(self):
        return array(self.__outflowEdge)

    def appendInflowEdge(self, nId):
        self.__inflowEdge.append(nId)

    def appendOutflowEdge(self, n_id):
        self.__outflowEdge.append(n_id)

    def appendIncomingNodes(self, n_id):
        self.__incomingNodes.append(n_id)

    def appendOutgoingNodes(self, n_id):
        self.__outgoingNodes.append(n_id)
