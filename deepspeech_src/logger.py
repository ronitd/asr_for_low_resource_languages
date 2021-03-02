import tensorflow as tf 

class TensorboardLogger(object): 
	def __init__(self, log_dir): 
		print("Creating log in {}".format(log_dir))
		self.writer = tf.compat.v1.summary.FileWriter(log_dir) 

	def scalar_summary(self, tag, value, step): 
		#print("Should be writing to log dir")
		summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]) 
		self.writer.add_summary(summary, step) 

