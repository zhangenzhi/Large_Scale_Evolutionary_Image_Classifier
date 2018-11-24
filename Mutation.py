import random
import copy
from DNA import DNA
import uuid
import Protocol_Buffer

class Mutation():
    pass



class AlterLearningRateMutation(Mutation):
    """Mutation that modifies the learning rate."""
    def mutate(self, dna):
        mutated_dna = copy.deepcopy(dna)
        # Mutate the learning rate by a random factor between 0.5 and 2.0,
        # uniformly distributed in log scale.
        factor = 2**random.uniform(-1.0, 1.0)
        mutated_dna.learning_rate = dna.learning_rate * factor
        return mutated_dna

class StrideMutation(Mutation):
    def mutate(self,dna):
        """mutates stride to 2 based number,like 1,2,4,8,16"""

        for conv_edge_id in self.find_allow_edge(dna):
            mutated_dna = copy.deepcopy(dna)
            if self.mutate_stride(conv_edge_id,mutated_dna):
                return mutated_dna
        return False

    def find_allow_edge(self,dna):

        conv_set = []
        for conv_edge_id in dna.edge:
            if dna.edge[conv_edge_id].type == 'conv':
                conv_set.append(conv_edge_id)
        # random choose one conv_edge
        random.shuffle(conv_set)

        return conv_set

    def mutate_stride(self,conv_edge_id,dna):
        # Mutate the stride by a random factor between 0.5 and 2.0,
        dna.edge[conv_edge_id].stride_scale= int(dna.edge[conv_edge_id].stride_scale*random.uniform(0.5,2))
        return True



class NumberOfChannelMutation(Mutation):
    def mutate(self,dna):

        for conv_edge_id in self.find_allow_edge(dna):
            mutated_dna = copy.deepcopy(dna)
            if self.mutate_numberofchannel(conv_edge_id,mutated_dna):
                return mutated_dna

        return False

    def find_allow_edge(self,dna):

        conv_set = []
        for conv_edge_id in dna.edge:
            if dna.edge[conv_edge_id].type == 'conv':
                conv_set.append(conv_edge_id)
        # random choose one conv_edge
        random.shuffle(conv_set)
        return  conv_set

    def mutate_numberofchannel(self,conv_edge_id,dna):
        # Mutate the depth factor by a random factor between 0.5 and 2.0,
        dna.edge[conv_edge_id].depth_factor = int(dna.edge[conv_edge_id].depth_factor*random.uniform(0.5,2))
        return True



class FilterSizeMutation(Mutation):
    def mutate(self,dna):

        for conv_edge_id in self.find_allow_edge(dna):
            mutated_dna = copy.deepcopy(dna)
            if self.mutate_filter(conv_edge_id,mutated_dna):
                return mutated_dna

        return False

    def find_allow_edge(self,dna):

        conv_set = []
        for conv_edge_id in dna.edge:
            if dna.edge[conv_edge_id].type == 'conv':
                conv_set.append(conv_edge_id)
        # random choose one conv_edge
        random.shuffle(conv_set)
        return  conv_set

    def mutate_filter(self,conv_edge_id,dna):
        # Mutate the filter size by a random factor between 0.5 and 2.0,
        dna.edge[conv_edge_id].filter_half_width= int(dna.edge[conv_edge_id].filter_half_width*random.uniform(0.5,2))
        dna.edge[conv_edge_id].filter_half_height=int(dna.edge[conv_edge_id].filter_half_height*random.uniform(0.5,2))
        return True




class AddEdgeMutation(Mutation):
    """ Add single edge to graph """

    def __init__(self,str):
        self.edge_type = str
        self.to_regex = 'output_mutable'

        self.from_regex = 'input_mutable'


    def mutate(self,dna):

        # self.edge_type = 'skip_connection'
        for from_vertex_id,to_vertex_id in self.vertex_pair_candidates(dna):
            mutated_dna = copy.deepcopy(dna)
            if self.mutate_structure(mutated_dna,from_vertex_id,to_vertex_id):
                return mutated_dna

        return False

    def find_allowed_vertex(self,dna,opposite):

        if opposite == 'output_mutable':
            """search vertex with output mutable"""

            vertex_set = set({})
            for vertex_id in dna.vertex:
                if dna.vertex[vertex_id].output_mutable:
                    vertex_set.add(vertex_id)
            return  vertex_set

        elif opposite == 'input_mutable':
            """search vertex with output mutable """

            vertex_set = set({})
            for vertex_id in dna.vertex:
                if dna.vertex[vertex_id].input_mutable:
                    vertex_set.add(vertex_id)
            return vertex_set

    def avoid_back_connection(self, dna,to_vertex_id):
        """
        Using Breadth First search for cycle.
        If one vertex  makes a cycle, it will find itself
        """

        edges = dna.vertex[to_vertex_id].edge_in

        if edges == {}:
            return {}

        vertex=set({})
        for edge_id in edges:
            ver = dna.edge[edge_id].from_vertex
            vertex.add(ver)

        new_vertex = set({})
        for vertex_id in vertex:
            [new_vertex.add(ver) for ver in self.avoid_back_connection(dna,vertex_id)]

        for item in vertex:
            new_vertex.add(item)

        return new_vertex



    def vertex_pair_candidates(self,dna):
        """ Yeild connectable vertex pair """


        from_vertex_ids_set =self.find_allowed_vertex(dna,self.to_regex)

        from_vertex_ids =[]
        for item in from_vertex_ids_set:
            from_vertex_ids.append(item)
        random.shuffle(from_vertex_ids)

        to_vertex_ids_set = self.find_allowed_vertex(dna,self.from_regex)

        to_vertex_ids = []
        for item in to_vertex_ids_set:
            to_vertex_ids.append(item)
        random.shuffle(to_vertex_ids)

        for to_vertex_id in to_vertex_ids:
            """avoid back connection"""
            disallowed_from_vertex = self.avoid_back_connection(dna,to_vertex_id)
            # print(disallowed_from_vertex)

            for from_vertex_id in from_vertex_ids:
                if from_vertex_id in disallowed_from_vertex:
                    continue
                """This pair does not generate cycle"""
                yield from_vertex_id,to_vertex_id
    def mutate_structure(self,dna,from_vertex,to_vertex):
        """Add edge to dna instance """
        edge_id = uuid.uuid4()
        edge_type = self.edge_type
        if dna.has_edge(from_vertex,to_vertex):
            return False
        else:
            dna.add_edge(from_vertex,to_vertex,edge_type,edge_id)
            return True

class RemoveEdgeMutation(Mutation):
    """This class will remove edge from edges"""

    def mutate(self,dna):
        """Randomly choose one edge in edges"""
        random_edge_id = random.choice(list(dna.edge.keys()))
        to_vertex_id = dna.edge[random_edge_id].to_vertex

        if dna.vertex[to_vertex_id].output_mutable == False:
            return False

        mutated_dna = copy.deepcopy(dna)
        """If the choosen edge is conv or identity"""
        if mutated_dna.edge[random_edge_id].type == 'conv' or mutated_dna.edge[random_edge_id].type == 'identity':

            from_vertex_id = mutated_dna.edge[random_edge_id].from_vertex
            to_vertex_id = mutated_dna.edge[random_edge_id].to_vertex

            mutated_dna.vertex[from_vertex_id].edge_out = mutated_dna.vertex[to_vertex_id].edge_out
            for edge_id in mutated_dna.vertex[from_vertex_id].edge_out:
                mutated_dna.edge[edge_id].from_vertex = from_vertex_id

            mutated_dna.vertex.pop(to_vertex_id)
            mutated_dna.edge.pop(random_edge_id)

        elif mutated_dna.edge[random_edge_id].type == 'skip_connection':

            from_vertex_id = mutated_dna.edge[random_edge_id].from_vertex
            to_vertex_id = mutated_dna.edge[random_edge_id].to_vertex

            mutated_dna.vertex[from_vertex_id].edge_out.remove(random_edge_id)
            mutated_dna.vertex[to_vertex_id].edge_in.remove(random_edge_id)
            mutated_dna.edge.remove(random_edge_id)

        return mutated_dna



class ResetWeight(Mutation):
    """
    Reset weight
    This part is not ready now
    """
    def mutate(self,dna):
        return  False

class Identity(Mutation):
    def mutate(self,dna):
        return dna



if __name__ == '__main__':
    pass
    # Reader = Protocol_Buffer.Protocol_Buffer()
    # dna_proto = Reader.Load_from_Disk()
    # a = DNA(dna_proto)
    #
    # print("---------------------Original------------------------")
    # print("Vertex ", a.vertex.keys())
    # print("Edge", a.edge.keys())
    #
    # addEdge = AddEdgeMutation()
    # a = addEdge.mutate(a)
    #
    # print("---------------------AddEdge------------------------")
    # print("Vertex ", a.vertex.keys())
    # print("Edge", a.edge.keys())
    #
    # removeEdge = RemoveEdgeMutation()
    # a = removeEdge.mutate(a)
    #
    # print("---------------------RemoveEdge------------------------")
    # print("Vertex ", a.vertex.keys())
    # print("Edge", a.edge.keys())



