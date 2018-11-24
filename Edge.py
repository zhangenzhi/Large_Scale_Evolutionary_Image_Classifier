import Protocol_Buffer

class Edge():
    """
    This class using edge protocol to initialize.
    Edge is connection between 2 vertex,so it has from vertex, to vertex.
    Edge is using to calculate convolution, or connect two vertex as identity or skip connection
    """
    def __init__(self,edge_proto):
        """
        1.Seting unique ID generated from uuid4()
        2.Choose it's type from identity,convolution,skip_connection
        """

        self.ID = edge_proto.ID

        self.from_vertex = edge_proto.from_vertex
        self.to_vertex = edge_proto.to_vertex

        if edge_proto.HasField('conv'):
            # In this case, the edge represents a convolution.
            # filter size is odd number
            # stride is exponential based 2, like 1,2,4,8,16
            self.type = 'conv'
            self.depth_factor = edge_proto.conv.depth_factor
            self.filter_half_width = edge_proto.conv.filter_half_width
            self.filter_half_height = edge_proto.conv.filter_half_height

            self.stride_scale = edge_proto.conv.stride_scale

            self.depth_precedence = edge_proto.depth_precedence
            self.scale_precedence = edge_proto.scale_precedence

        elif edge_proto.HasField('identity'):
            self.type = 'identity'

        elif edge_proto.HasField('skip_connection'):
            self.type = 'skip_connection'


    def to_proto(self):

        """
        This function convert a vertex instance to protocol buffer form
        """
        edge_proto = Protocol_Buffer.Edge_Protocol()

        edge_proto.from_vertex = self.from_vertex
        edge_proto.to_vertex = self.to_vertex

        if self.type == 'conv':
            edge_proto.type = 'conv'
            edge_proto.conv.depth_factor = self.depth_factor
            edge_proto.conv.filter_half_height = self.filter_half_height
            edge_proto.conv.filter_half_width = self.filter_half_width

            edge_proto.conv.stride_scale = self.stride_scale

        elif self.type == 'identity':

            edge_proto.type = self.type

        edge_proto.depth_precedence = self.depth_precedence
        edge_proto.scale_precedence = self.scale_precedence

        return  edge_proto

    def depth_out(self,depth):
        """
        control the depth(output channel of convolution layer)
        """
        return  depth*self.depth_factor

    def tensor_hold(self,tensor):
        """
        Pass the tensor from from_vertex to to_vertex of skip_connection
        """
        self.tempt_tensor = tensor

if __name__ == '__main__' :
    pass