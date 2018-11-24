
from Protocol_Buffer import Protocol_Buffer
import random
from DNA import DNA
from Model import Model
import Mutation
import tensorflow.examples.tutorials.mnist.input_data as input_data

def takele0(ele):
    return ele[0]

class Worker():
    """

    Worker choose two dna protocols from Protocol_Buffer.
    Selects the dna with better fitness to mutate,add kill the bad dna in the Protocol Buffer.
    Mutated dna can be seen as child,and after mutate,we calculate the fitness for child.
    Save child to Protocol Buffer and use it's fitness as name.

    Next Version:
        I would like to run multiple works asynchronous on one computer but multi-GPUs.
        Further more, I'd like run worker on different PC.

    """
    mnist = input_data.read_data_sets("MNIST", one_hot="True")
    iteration_limit = 2
    def __init__(self,Protocol_Buffer):
        """
        1.Build a mutation table which contains all the allowed mutation
        2.connects to Protocol Buffer which contains all DNA protocol
        """
        #Build mutation table first
        alertLearningRate = Mutation.AlterLearningRateMutation()
        alertStride = Mutation.StrideMutation()
        alerFiltersize = Mutation.FilterSizeMutation()
        alertFilterNumbers = Mutation.NumberOfChannelMutation()
        addConv =Mutation.AddEdgeMutation('conv')
        insertOnetoOne = Mutation.AddEdgeMutation('identity')
        addSkipConnection = Mutation.AddEdgeMutation('skip_connection')
        Identity = Mutation.Identity()
        removeSkipConnection = Mutation.RemoveEdgeMutation()
        removeConv = Mutation.RemoveEdgeMutation()
        resetWeight = Mutation.ResetWeight()



        self.mutation_list=[alerFiltersize,alertFilterNumbers,alertStride,
                            alertLearningRate,resetWeight,
                            addConv,addSkipConnection,insertOnetoOne,
                            removeConv,removeSkipConnection,Identity]
        self.pb = Protocol_Buffer


    def Fitness(self,individual):
        """
        1.Calculate the fitness to this individual
        2.Return fitness
        """

        set_individual = Model(dna = individual)
        fitness = set_individual.model(mnist = self.mnist)
        return fitness

    def Selection(self):
        """
        1.Choose two DNA Protocol from Protocol Buffer randomly
        2.kill the one with lower fitness
        3,return the dna protocol with high fitness
        """
        fitness_1, individual_1 = self.pb.Load_from_Disk()
        fitness_2, individual_2 = self.pb.Load_from_Disk()
        if fitness_1>fitness_2:
            self.pb.Killer(individual_2.ID)
            selected_individual = individual_1
        else:
            self.pb.Killer(individual_1.ID)
            selected_individual = individual_2

        return selected_individual


    def BreedChild(self,selected_individual):
        """
        1.Choose one kind of mutation in the mutation table,and make a child
        2.calculate the fitness for this child
        3.Save this child to Protocol Buffer

        """

        Mutate = random.choice(self.mutation_list)
        dna = DNA(selected_individual)
        child = Mutate.mutate(dna)

        while child == False:
            Mutate = random.choice(self.mutation_list)
            child = Mutate.mutate(DNA(selected_individual))

        fitness = self.Fitness(child)
        self.pb.Save_to_Disk(fitness,child.to_proto())


    def Run(self):
        turn = 0
        while turn != self.iteration_limit:
            selected_individual= self.Selection()
            self.BreedChild(selected_individual)
            turn += 1

if __name__ == '__main__':
    pb = Protocol_Buffer()
    worker = Worker(pb)
    worker.Run()



