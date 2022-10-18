import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# create a tensor with the shape 2x3 with a data type float 32
x = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3), dtype=tf.float32)


# create a tensor with the shape 3x3 filled with 1
x = tf.ones((3, 3))

# create a tensor with the shape 3x3 filled with 0
x = tf.zeros((3, 3))


# create a tensor with the shape 3x3 where the diagonal line is 1
x = tf.eye(5)


# create a tensor with the shape 3x3 filled with random values
x = tf.random.uniform((3, 3), minval=0, maxval=1)


# create a tensor filled with 0-9
x = tf.range(10)


# cast a tensor, change its data type
x = tf.cast(x, dtype=tf.float32)


# add 2 elements, will return [1 + 4, 2 + 5, 3 + 6]
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

z = tf.add(x, y)
z = x + y


# substract 2 elements, will return [1 - 4, 2 - 5, 3 - 6]
z = tf.subtract(x, y)
z = x - y

# divide 2 elements, will return [1 / 4, 2 / 5, 3 / 6]
z = tf.divide(x, y)
z = x / y

# multiply 2 elements, will return [1 * 4, 2 * 5, 3 * 6]
z = tf.multiply(x, y)
z = x * y

# multiply 2 elements and return the sum
z = tf.tensordot(x, y, axes=1)

# power all the elements in the tensor by 3
z = x ** 3

# multiply a matrix, column length of x must match row length of y
x = tf.random.normal((2, 2))
y = tf.random.normal((2, 2))

z = tf.matmul(x, y)
z = x @ y


# slicing, indexing
x = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])

x[0] # access first column
x[:, 0] # access all the columns, but only the first element in these columns
x[0, :2] # access the first column and get the first 3 elements
print(x[0, 1]) # access the first column and get the 2nd element


# reshaping
x = tf.random.normal((2, 3))

x = tf.reshape(x, (6))
x = tf.reshape(x, (3, 2))
x = tf.reshape(x, (-1, 2))  # like (3, 2)


# convert to a numpy array
x = tf.random.normal((2, 3))

x = x.numpy()

# convert back to a tensor
x = tf.convert_to_tensor(x)


# a number is not required for a tensor
x = tf.constant(["test"])


# varible. a tensor varible can be modified
x = tf.Variable([1, 2, 3])