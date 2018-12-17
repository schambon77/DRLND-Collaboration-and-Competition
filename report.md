[//]: # (Image References)

[image1]: https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/scores.png "Plot of Rewards"
[image2]: https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/training_trace.PNG "Output trace during training"

# Project 3: Collaboration and Competition

## Technical Details

The solution is based in most parts on the source code provided during the [Udacity lab on Multi-Agent DDPG](https://classroom.udacity.com/nanodegrees/nd893/parts/ec710e48-f1c5-4f1c-82de-39955d168eaa/modules/89b85bd0-0add-4548-bce9-3747eb099e60/lessons/a6347d5b-55f0-45cd-bab2-308f877d79a7/concepts/475be8a3-68d3-42ed-8605-90c905d88ab0),
 which is inspired by the work described in the paper published by Lowe et al [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

### Learning Algorithm

The goal is to train 2 agents with each a separate actor but a shared critic and experience buffer. The shared critic 
and experience replay buffer aims at stabilizing the training of both agents.
During training, both agents use their separate actor network to predict the best action to take based on 
their respective state. Some decaying noise is used in order to favour exploration early in the training, and then
focus on exploitation once the networks start to converge.

Actions are passed to the environment, and the overall experience tuples are stored in a shared buffer.

Once enough experience tuples have been stored, a batch of size **128** is sampled from the buffer for each agent in order
to update the shared critic. A discount value of **0.95** and a learning rate of **1.0e-3** are used for both agents.

A soft update of all target networks is finally applied with a tau value of **0.1**.

Scores for both agents are accumulated separately over each episode, with the maximum of the 2 kept in the main score 
queue of the last 100 in order to compute the overall score to be brought above **+0.5**.

#### Networks

The generic class `Network` is used for all 6 networks used in this work.

##### Actor

The actor network structure is defined as:

| Input Size        | Layer           | Output Size |
| ------------- |:-------------:| -----:|
| 24       |  Fully Connected    | 256 |
| 256      | RELU     |   256 |
| 256 | Fully Connected     |    128 |
| 128      | RELU     |   128 |
| 128 | Fully Connected     |    2 |
| 2 | tanh     |    2  |

The input size of the first layer, 24, corresponds to the size of an agent observation, and the output size of the 
last layer, 2, corresponds to the size of the action space.

##### Critic

The critic network structure is defined as:

52, 256, 128, 1

| Input Size        | Layer           | Output Size |
| ------------- |:-------------:| -----:|
| 52       |  Fully Connected    | 256 |
| 256      | RELU     |   256 |
| 256 | Fully Connected     |    128 |
| 128      | RELU     |   128 |
| 128 | Fully Connected     |    1 |

The input size of the first layer, 52, corresponds to the size of both agent observations, 24 * 2, plus both agent actions,
2 * 2. The output size of the last layer is typically 1 for a critic network that approximates the action-value function.

#### Training

Here is the output trace generated by the main.py script during training.

![Output trace during training][image2]

### Plot of Rewards

The training reaches the acceptance criteria after 755 episodes.

![Plot of Rewards][image1]


### Ideas for Future Work

Ideas for future work include improving the efficiency of the training, as well as simplifying the source code.
We can note that due to the environment symmetry and lack of competition in the reward structure, a simpler single DDPG 
agent might be able to solve the environment faster.