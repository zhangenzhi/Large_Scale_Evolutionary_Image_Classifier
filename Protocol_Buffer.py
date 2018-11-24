import pickle
import uuid
import os
import random
import numpy as np
import re

class Protocol_Buffer():
    """
    This class initialize the enviroment and manage all the operation of dna protocol.
    The operation,For example,like save to local disk,load from disk

    In next version, I'd like to using this class to create a file share system
    to communicate with other computer.
    """

    def __init__(self):
        """
        Initialize, setting the population size to 50(In paper their setting is 250),
        but here is very easy to change
        """
        self.population_limit = 100
        self.population_size = 50
        self.InitialPopulation()

    def Save_to_Disk(self,fitness,dna_proto):
        """
        Save dna protocol to disk in Protocol_Buffer.
        Name = "ID + .pickel"
        """
        dna_proto.fitness = fitness
        with open('./Protocol_Buffer/'+str(dna_proto.ID)+".pickel", 'wb') as p:
            pickle.dump(dna_proto, p)

    def Load_from_Disk(self):
        """
        Load dna protocol randomly from Protocol_Buffer.
        Return fitness and dna protocol
        """

        filename = random.choice(os.listdir('./Protocol_Buffer'))
        with open('./Protocol_Buffer/'+filename,'rb') as r:
            dna_proto = pickle.load(r)

        fitness = dna_proto.fitness

        return fitness,dna_proto

    def InitialPopulation(self):
        """
        Initial Protocol_Buffer.
        The initial fitness is about 0.1,because the number of image class is 10 ,
        which means the probability is of right choice is 0.1,
        distribution is seen as uniform from 0.8 to 1.2
        """
        population = []
        for i in range(self.population_size):
            population.append(DNA_Protocol())

        for protocol in population:
            fitness = np.random.uniform(0.08,0.12)
            self.Save_to_Disk(fitness,protocol)


    def Killer(self,ID):
        """
        Using to remove the dna protocol with low fitness
        """
        filename = str(ID) + ".pickel"
        if os.path.exists("./Protocol_Buffer/"+filename):
            os.remove("./Protocol_Buffer/"+filename)
        else:
            print("The file does not exist")


class DNA_Protocol():
    """
    This class is the encoded information which hold the full information to build a model.
    The topology of DNA Protocol is a Graph.So it hold it's vertices and edges information
    Also holds learning rate to build model
    """

    def __init__(self):
        """
        Initial a framework, 2 vertex and one edge between.
        2 vertex control input and output and they can't be mutate, they are linear
        1 edge is identity, which meas just pass tensor
        """
        init_vertex_proto_input = Vertex_Protocol()
        init_vertex_proto_input.type = 'linear'
        init_vertex_proto_output = Vertex_Protocol()
        init_vertex_proto_output.type = 'linear'
        init_edge = Edge_Protocol()
        init_edge.type = 'identity'

        init_vertex_proto_input.output_mutable = True
        init_vertex_proto_output.input_mutable = True


        init_vertex_proto_input.edge_out.add(init_edge.ID)
        init_vertex_proto_output.edge_in.add(init_edge.ID)
        init_edge.from_vertex = init_vertex_proto_input.ID
        init_edge.to_vertex = init_vertex_proto_output.ID


        self.vertex_proto = {init_vertex_proto_input.ID:init_vertex_proto_input,
                             init_vertex_proto_output.ID:init_vertex_proto_output}
        self.edge_proto = {init_edge.ID:init_edge}
        self.learning_rate = 0.1
        self.fitness = 0
        self.ID = uuid.uuid4()

class Vertex_Protocol():
    """
    This class is the encoded information which can create a Vertex
    Vertex is connection between  edges,so it has input edge, output edge.
    Vertex is using to calculate unlinear result, like bn_relu,or linear
    """
    def __init__(self):
        self.edge_in = set({})
        self.edge_out = set({})

        self.ID = uuid.uuid4()

        self.type = 'linear'

        self.input_mutable = False
        self.output_mutable = False
        self.property_mutable = False

    def HasField(self,type):

        if self.type == type:
            return True

        return False

class Conv():
    """
    Conv holds the whole information to config one convolution layer

    """

    def __init__(self):
        self.depth_factor= 1
        self.filter_half_width = 1
        self.filter_half_height = 1
        self.stride_scale = 0
        

class Edge_Protocol():
    """
    This class is the encoded information which can create a Vertex
    Edge is connection between 2 vertex,so it has from vertex, to vertex.
    Edge is using to calculate convolution, or connect two vertex as identity or skip connection
    """

    def __init__(self):

        self.from_vertex = None
        self.to_vertex = None

        self.ID = uuid.uuid4()
        self.type = 'identity'
        self.conv = Conv()

        self.depth_precedence = 1
        self.scale_precedence = 1


    def HasField(self,type):
        if self.type == type:
            return True
        return False

if __name__=='__main__':
    pass