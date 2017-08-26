import tensorflow as tf

sess = tf.Session()

m = tf.Variable([0.3])		# Represents a linear grah of y = 0.3x - 0.3
c = tf.Variable([-0.3])
x = tf.placeholder(tf.float32)

linear_model = m*x + c

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
# [ 0.          0.30000001  0.60000002  0.90000004]

# Now to test our network.
# We create a Y placeholder to hold our test values (I think)

y = tf.placeholder(tf.float32)

# Now we need to create a function to measure loss.
# Loss is the difference between the value that the algorithm produces
# and the value that we want (or the training data wants).
# The standard loss function for linear grahs is ∑((actual_results - desired_results)²)

differences_squared = tf.square(linear_model - y)
loss = tf.reduce_sum(differences_squared)		# tf.reduce_sum() adds up together all the items in a tensor. It's like 'n = 0; for i in tensor: n += i;'

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# The loss turns out to be 23.66

# Suffice to say that I am not too confident with this toic and there may be mistakes.
