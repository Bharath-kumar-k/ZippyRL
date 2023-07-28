from node import *


class Edge:
    def __init__(self, edgeId=None, fromNode=None, toNode=None):
        self.__edgeId = edgeId
        self.__fromNode = fromNode
        self.__toNode = toNode
        self.__fromNodeId = fromNode.nodeId()
        self.__toNodeId = toNode.nodeId()
        self.__edgeCost = None
        self.__edgeDist = None
        self.__edgeAng = None

    def __eq__(self, other):
        return True if self.__edgeId == other.__edgeId else False

    def edgeId(self):
        return self.__edgeId

    def edgePos(self, returnWithEdgeId=False):
        return (self.__edgeId, array([self.__fromNode.nodePos(), self.__toNode.nodePos()])) \
            if returnWithEdgeId else (array([self.__fromNode.nodePos(), self.__toNode.nodePos()]))

    def fromNode(self):
        return self.__fromNode

    def toNode(self):
        return self.__toNode

    def fromNodeId(self):
        return self.__fromNodeId

    def toNodeId(self):
        return self.__toNodeId

    def edgeCost(self):
        return self.__edgeCost

    def edgeCost(self, weight):
        self.__edgeCost = weight

    def edgeDist(self):
        return self.__edgeDist

    def edgeDist(self, distance):
        self.__edgeDist = distance

    def edgeAng(self):
        return self.__edgeAng

    def edgeAng(self, angle):
        self.__edgeCost = angle