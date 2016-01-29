#!/usr/bin/env python

# This is an attempt to parallelize the training of a SDM on TensorFlow.
# https://en.wikipedia.org/wiki/Sparse_distributed_memory#Implementation
import tensorflow as tf
import time 



# Stopped while still deciding what data structure is best for 10K bitstrings.





# An SDM can be thought of as a 2 layer NN.
# http://science.slc.edu/~jmarshall/courses/2002/fall/cs152/lectures/SDM/sdm.html

# 
# http://pendicular.net/cbvs.php
#   "The memory examples described by Kanerva in 1988 with one million hard locations require about ten billion bytes of high-speed storage, which is feasible today."

# The first layer nodes are called address decoders. This array is by def of SDM an immutable array (useful feature for parallel model)
# rand_array = np.random.rand(1024, 1024) # gives a shape=(1024, 1024) array  
# maybe address_decoders_addresses = np.random.rand(one_million_a_ds, 128) # a million 128 byte addresses for the address decoders to each center on for hamming
# JFT-TODO: how to in TensorFlow mark immutable, just like my_numpy_array.setflags(write=False)
#   tf.constant( some_128_byte_random )
#   tf.uint8
address_decoders_addresses = tf.tensorYA(1M address decoders) # 1M elements each a 10K-bit char[], sized 1280c. That's about a Gig?


# The second layer nodes are called storage registers.
# This is where the model accumulates the results of the training.
# The elements in each storage register's content array are integers, not bits like is the case with address decoders.
storage_registers = tf.constant( [0, 1,  1M]) # tensor of 1M with each element a 10K array of integers

# An SDM has 2 input 1-D arrays: address and data. Those inputs are bits, 1000 of them each.
# JFT-TODO: there has to be many more efficient ways to encode very long (~10K) binary numbers
# Looks like char[] might be good for C interfacing (i.e. format type s of size 128c. 128 x 8bit = 1024 i.e. just over 1K. But then need 10 of those. So, actually, 1280c?)
#   https://docs.python.org/3/library/struct.html
# Or SciPy has ndarray
#   http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
#     ndarray.strides Tuple of bytes to step in each dimension when traversing an array.
#   So my_sdm.strides = 1280 # times 8 bits is close-enough-to 10,000 bits
#   http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
# Bitarray might be useful
#   https://pypi.python.org/pypi/bitarray/
#   Bitwise operations: &, |, ^, &=, |=, ^=, ~
#   Packing and unpacking to other binary data formats, e.g. numpy.ndarray, is possible.
# bitstring sounds nice too
#   http://stackoverflow.com/a/2555287/4669056
my_sdm = ??
my_sdm.ddress_input = tf.constant([1, 0, 0, 1,  ...x10000])
my_sdm.data_input = tf.constant(  [0, 1, 1, 1,  ...x10000])

start_time = time.clock()
train_sdm(my_sdm)
stop_time = time.clock()
print 'training time: {}'.format( stop_time - start_time )



# The first layer's job is to decide which nodes in the second layer will be involved in the read xor write.
# Need to elect the 3 (or so) address decoders which have highest ~"hamming distance" score on match-to-address-input
#   1. Each address decoder evaluates it's match
#   2. They elect highest scorers to fire to level 2 of network
# JFT-TODO:
#   Each address decoder could be running in parallel (GPU a set of them, different sets on different hardware).
#   But how does that
def train_sdm(a_sdm):
    # While trainings, set to "write":
    is_in_training_mode = True

    # this is bit by bit compare of two 1-D array's of bits (about 1000) and sum hits:
    # JFT-TODO: so this isn't best as a matrix mult
    address_decoder_hammings = tf.matmul( address_input, a_sdm.address_decoders_addresses )

    with tf.Session() as sess:
        print sess.run(a_sdm)
        
    # closest_address_decoders is a short list, usually 3 adress decoders that just so happen to be closest to the address_input
    closest_address_decoders = select_closest( address_decoder_hammings )

    for address_decoder in closest_address_decoders:
        if is_in_training_mode:
            address_decoder.second_layer_node.write_data_input_to_storage_register()
        else:
            address_decoder.second_layer_node.compare_data_input_to_storage_register_for_recognition()

    return False # return what? Dunno.
        

        
# An optimation for parallel computing?
# Address decoders and the nodes they can trigger should all be on the same GPU.
# So, address decoding goes on in parallel, the top 3 nodes' score is sent over net for voting on global 3 highest, but while that is being
# computed remotely, decide what would be written or read for all GPU-local 2nd layer nodes.
# If voting comes back as "yes, one (or 2, or 3) of your address decoders won" then save the layer 2 just precomputed.
# If voting comes back as "none of yours" then just discard all those local layer 2 precomputes that were done while waiting for global vote answer.

# if some bits of the address vector are known to correlate with some feature that partitioning can happen on that would be great.
# Then could just map first few bits to machine-where-corresponding-address-decoders-is
# That is how to load balance \

# Note: the "weights" or values of the address decoders are randomly set bits that are set only once, at network initialization.
# So that (an immutable array 1 million elements long, each element a thousand-bit word) can be replicated across network, each machine having that all the time.
# The only thing that changes is the "weigths" in the second layer, the storage registers. This is the model that needs to be shared across network. That too
# is an array 1 million elements long, each element a thousand-bit word. This one is of course mutable.

# So this is a data parallelism compute network architecture.
# - Each machine is training the SDM locally.
# - Because writing to storage registers is done very sparsely and because high-dim of memory_datum that is written
#   - b/c the above 2, can update distributed model in hogwild fashion? Hopefully.
# Just what are the storage registers doing? Accumulating counts.
#   - Those will not overwrite each other negatively during the model merges. Just add/subtract to register's cells and that's the new model.
