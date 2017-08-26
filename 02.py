# Here, we write a network which can plot the y-axis of a linear graph
# with the equasion y = mx + c. In this case the equasion for the graph
# is y = 2x + 3.

import tensorflow as tf

sess = tf.Session()

# Unlike constants, tensorflow variables can be changed.
# However, they need to be initialized before the network is run.

m = tf.Variable([2.0])		# We give it a rank [1] tensor for some reason.
c = tf.Variable([3.0])
x = tf.placeholder(tf.float32)	# x is just a placeholder. We give x a value every time the tensor is run.
				# It's a bit like giving an argument to the neural network

linear_model = m * x + c	# Here we create our node to calculate the y value. '*' and '+' implies tf.add() and tf.multiply(), respectivley.

# Initialize the variables; we need to actually *run* the initializer too.

init = tf.global_variables_initializer()
sess.run(init)

# Now we can run our neural network for multiple values of x on the graph
sess.run(linear_model, {x: [0, 1, 2, 3]})
