# SAC
Update 2021-03-14: 

Don't consider V network(it can be replaced by Q network)

1. I just apply action without scaling(the output action value of actor network has range from 0 to 1. so this value should be scaled)
2. I adopt replay buffer into 2 method. One is training all the replay buffer or part of the replay buffer(with batch_size). But i think it is not important. Both of them need enough training epoch when training.
