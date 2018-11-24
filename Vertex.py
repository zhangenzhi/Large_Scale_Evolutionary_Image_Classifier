import Protocol_Buffer

class Vertex():
    """
    This class using vertex protocol to initialize.
    Vertex is connection between  edges,so it has input edge, output edge.
    Vertex is using to calculate unlinear result, like bn_relu,or linear
    """

    def __init__(self,vertex_proto):

        """
        1.Seting unique ID generated from uuid4()
        2.Leakiness is using for relu
        """

        self.ID = vertex_proto.ID
        self.leakiness = 0.1

        self.edge_in = set(vertex_proto.edge_in)
        self.edge_out = set(vertex_proto.edge_out)

        if vertex_proto.HasField('relu_bn'):
            self.type = 'relu_bn'
        elif vertex_proto.HasField('linear'):
            self.type = 'linear'

        self.input_mutable = vertex_proto.input_mutable
        self.output_mutable = vertex_proto.output_mutable
        self.property_mutable = vertex_proto.property_mutable

    def to_proco(self):
        """
        This function convert a vertex instance to protocol buffer form
        """
        vectex_proto = Protocol_Buffer.Vertex_Protocol()

        vectex_proto.edge_in = self.edge_in
        vectex_proto.edge_out = self.edge_out

        vectex_proto.type = self.type

        vectex_proto.input_mutable = self.input_mutable
        vectex_proto.output_mutable = self.output_mutable
        vectex_proto.property_mutable = self.property_mutable

        return vectex_proto

if __name__ == '__main__' :
    a = Protocol_Buffer.Vertex_Protocol()

    v = Vertex(a)
    print(v.type)