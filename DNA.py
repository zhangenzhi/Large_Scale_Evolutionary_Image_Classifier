import Protocol_Buffer
import random
from Protocol_Buffer import  Edge_Protocol
from Protocol_Buffer import  Vertex_Protocol
from Vertex import Vertex
from Edge import Edge

class DNA():
    """
    This class using dna protocol to instance a DNA which can be used to generated a Model
    The topology of DNA is a Graph.So it hold it's vertices and edges information
    Also holds learning rate to build model, and Addedge function to add different type edges
    """

    def __init__(self, dna_proto):

        self.learning_rate = dna_proto.learning_rate
        self.weight_decay_rate = 0.0001
        self.ID = dna_proto.ID

        self.vertex = {}  # String vertex ID to ‘Vertex‘ instance.
        for vertex_id in dna_proto.vertex_proto:
            self.vertex[vertex_id] = Vertex(vertex_proto=dna_proto.vertex_proto[vertex_id])

        self.edge = {}  # String edge ID to ‘Edge‘ instance.
        for edge_id in dna_proto.edge_proto:
            self.edge[edge_id] = Edge(edge_proto=dna_proto.edge_proto[edge_id])

    def to_proto(self):
        """
        Returns this instance in protocol buffer form.
        """
        dna_proto = Protocol_Buffer.DNA_Protocol()
        dna_proto.learning_rate = self.learning_rate

        return dna_proto

    def has_edge(self,from_vertex_id,to_vertex_id):

        if to_vertex_id in self.vertex[from_vertex_id].edge_out:
            return True

        return False

    def add_edge(self,from_vertex_id, to_vertex_id, edge_type, edge_id):
        """
        Adds an edge to the DNA graph, ensuring internal consistency.
        The parameters are decided by Addedgemutaion in Mutation.py
        """
        # ‘EdgeProto‘ defines defaults for other attributes.

        old_edge = 0
        for old_edge_id in self.vertex[from_vertex_id].edge_out:
            if self.edge[old_edge_id].type != 'skip_connection':
                old_edge = old_edge_id

        if edge_type == 'conv':

            """
            Insert convolution layer in two step:
            1. Insert a new vertex between from_vertex and the identity or convolution edge.
            2. Insert a convolution edge between from_vertex and new vertex
            3. add the new vertex and new edge to dna graph 
            
            """
            #1.add a new vertex first
            new_vertex_proto = Vertex_Protocol()
            new_vertex_proto.type = random.choice(['relu_bn', 'linear'])
            new_vertex = Vertex(new_vertex_proto)

            #2.add conv edge
            new_edge_proto= Edge_Protocol()
            new_edge_proto.type = 'conv'
            new_edge = Edge(new_edge_proto)
            new_edge.type = edge_type
            new_edge.ID = edge_id

            #config input and output for new_edge and new_vertex

            new_edge.from_vertex=from_vertex_id
            new_edge.to_vertex = new_vertex.ID

            new_vertex.input_mutable = True
            new_vertex.output_mutable = True
            new_vertex.edge_in.add(new_edge.ID)
            new_vertex.edge_out.add(old_edge)

            self.edge[old_edge].from_vertex = new_vertex.ID


            self.vertex[new_vertex.ID] = new_vertex
            self.edge[new_edge.ID] = new_edge


        elif edge_type == 'identity':

            """
            Insert convolution layer in two step:
            1. Insert a new vertex between from_vertex and the identity or convolution edge.
            2. Insert an identity edge between from_vertex and new vertex
            3. add the new vertex and new edge to dna graph 
            
            """
            # 1.add a new vertex first

            new_vertex_proto = Vertex_Protocol()
            new_vertex_proto.type = 'linear'
            new_vertex = Vertex(new_vertex_proto)

            # 2.add identity edge

            new_edge = Edge(Edge_Protocol())
            new_edge.type = edge_type
            new_edge.ID = edge_id

            new_vertex.input_mutable = True
            new_vertex.output_mutable = True
            new_vertex.edge_in.add(new_edge.ID)
            new_vertex.edge_out.add(old_edge)

            self.edge[old_edge].from_vertex = new_vertex.ID

            self.vertex[new_vertex.ID] = new_vertex
            self.edge[new_edge.ID] = new_edge

        elif edge_type=='skip_connection':

            """
            Add a skip connection between from_vertex and to_vertex
            """
            self.vertex[from_vertex_id].edge_out.add(edge_id)
            self.vertex[to_vertex_id].edge_in.add(edge_id)

            edge_proto = Protocol_Buffer.Edge_Protocol()
            edge_proto.type = edge_type
            edge_proto.ID = edge_id
            edge_proto.from_vertex = from_vertex_id
            edge_proto.to_vertex = to_vertex_id
            self.edge[edge_id] = Edge(edge_proto)


if __name__ == '__main__':
    pass



