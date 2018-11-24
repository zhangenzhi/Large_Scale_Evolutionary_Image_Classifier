
import tensorflow.contrib.slim as slim
import tensorflow as tf

class Model():
    """
    This class uses the information in DNA(Graph) to fold into a real Neural Network
    """

    def __init__(self,dna):
        """
        initialize the paramters used to build model
        """
        self.dna = dna
        self.batch_norm_epsilon = 0.1
        self.is_training = False

        self.batch_size = 100

    def model(self,mnist):
        """
        1.Folding DNA to a Neural Network
        2.Train this model using stochastic gradient descent method
        3. Using a Full connection layer to discriminate(Later I will replace this by Global pooling layer)
        """

        batch_nums = mnist.train.num_examples // self.batch_size

        #Hold input
        with tf.name_scope("Input"):
            x = tf.placeholder(tf.float32,shape=[None,784],name="TrainingData")
            y = tf.placeholder(tf.float32,shape=[None,10],name="TrainingLabel")
            with tf.name_scope("Image"):
                x_image = tf.reshape(x,[-1,28,28,1])

        # find initial vertex
        init_vertex =0
        for vertex_id in self.dna.vertex:
            if self.dna.vertex[vertex_id].input_mutable == False:
                init_vertex = self.dna.vertex[vertex_id]
                break

        tensor = x_image

        """Calculate the tensor from graph"""
        result_of_convbone= self.compute_graph(tensor,init_vertex)

        #FC layer
        fc_flatten = tf.layers.flatten(result_of_convbone)
        with tf.name_scope("FC_Layer"):
            fc1= tf.layers.dense(fc_flatten,units=32,activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1,units=64,activation=tf.nn.relu)
            fc3 = tf.layers.dense(fc2, units=10, activation=tf.nn.relu)

            with tf.name_scope("Softmax"):
                prediction = tf.nn.softmax(fc3)

        """
        Training
        1.loss function is defined by softmax cross_entropy
        2.Using stochastic Gradient descent method to optimise
        """

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.dna.learning_rate)
        with tf.name_scope("Train"):
            train = optimizer.minimize(loss)
        with tf.name_scope("Accuracy"):
            correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

        #save model
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            acc =0
            max =0
            for epoch in range(100):
                for batch in range(batch_nums):
                    x_train, y_train = mnist.train.next_batch(self.batch_size)
                    sess.run(train, feed_dict={x: x_train, y: y_train})

                acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                if acc>max:
                    max=acc
                print(acc)

            #save graphical network

            writer = tf.summary.FileWriter('logs/', sess.graph,filename_suffix=str(max))
            save_path = saver.save(sess, "save_model/"+str(max)+"_model.ckpt")

            return acc


    def compute_graph(self,tensor,vertex):
        """
        Calculate the graph when tensor flows in on the Convolution Backbone
        """

        while(True):

            for edge_in_id in vertex.edge_in:
                if self.dna.edge[edge_in_id].type == 'skip_cpnnection':
                    tensor += self.dna.edge[edge_in_id].tempt_tensor
            tensor = self.compute_vertex_nonlinearity(tensor, vertex)
            if vertex.output_mutable == False:
                break
            temp_tensor = tf.constant(0)
            new_vertex_id = 0
            for edge_out_id in vertex.edge_out:
                if self.dna.edge[edge_out_id].type == 'skip_connection':
                    self.dna.edge[edge_out_id].tensor_hold(tensor)
                elif self.dna.edge[edge_out_id].type == 'conv'or self.dna.edge[edge_out_id].type == 'identity':
                    temp_tensor = self.compute_edge_connection(tensor,self.dna.edge[edge_out_id])
                    new_vertex_id = self.dna.edge[edge_out_id].to_vertex
            tensor = temp_tensor
            vertex = self.dna.vertex[new_vertex_id]

        return tensor

    def compute_vertex_nonlinearity(self,tensor,vertex):
        """
        Compute tensor when flows into a vertex
        """

        if vertex.type == 'linear':
            pass
        elif vertex.type == 'relu_bn':
            tensor = slim.batch_norm(inputs=tensor,decay=0.9,center=True,scale=True,
                                     epsilon=self.batch_norm_epsilon,
                                     activation_fn=None,updates_collections=None,
                                     is_training=self.is_training,scope='batch_norm')

            tensor = tf.maximum(tensor,vertex.leakiness*tensor * tensor,name='relu')
        return tensor

    def compute_edge_connection(self,tensor,edge):
        """
        Compute tensor when flows into an edge
        """

        if edge.type == 'identity' :
            pass
        elif edge.type == 'conv':
            _,_,_,depth = tensor.get_shape()
            depth_out = edge.depth_out(depth)


            stride = 2**edge.stride_scale

            weights_initializer = slim.variance_scaling_initializer(factor=2.0,uniform=False)
            weights_regularizer = slim.l2_regularizer(self.dna.weight_decay_rate)

            print(depth_out,edge.filter_half_height*2,stride)
            tensor = slim.conv2d(inputs=tensor,num_outputs= depth_out,
                        kernel_size=[edge.filter_half_width*2, edge.filter_half_height*2],
                        stride = stride, weights_initializer = weights_initializer,
                        weights_regularizer = weights_regularizer,biases_initializer= None,
                        activation_fn = None,padding='SAME')

        return tensor


if __name__ == '__main__':
    pass

