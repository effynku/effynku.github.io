# Deep Reinforcement Learning for Page-wise Recommendations

## Abstract

Key problems for interaction for recommendations:

1. how to update recommending strategy according to user's real-time feedback
2. how to generate a page of items with proper display which pose tremendous challenges to traditional recommender systems

A principled approach: jointly generate a set of complementary items and corresponding strategy to display them in a 2-D page; optimize a page of items with proper display based on real-time feedback using deep reinforcement learning

### 1.1 Real-time Feedback

Recommendation procedure = sequential interactions between users and the recommender agent

Reinforcement Learning: automatically learn the optimal recommendation strategies

Advantages:

1. continuously update strategies based on user's real-time feedback during interactions, unitil the system converges to the optimal strategy that generates recommendations best fitting user's dynamic preferences
2. the optimal strategy is made by maximizing the expected long-term cumulative reward from users

### 1.2 Page-wise Recommendations

1. generate a set of diverse and complementary items
2. form an item display strategy to arrange the items in a 2-D page that can lead to maximal reward

*2-D grid rather than 1-D list/users partition the 2-D page into chunks and browse the chunk they prefer more.

### 2. Framework

Model the recommendation task as Markov Decision Process(MDP) and leverage Reinforcement Learning:

$MDP(S,A,P,R,\gamma)$: 

State space $S$: user's current preference, generated based on user's browsing history

Action space $A$: $a = \{a^1, …, a^M\} \in A$ action to recommend a page of M items to a user based on current state s

Reward $R$: after the RA takes an action a at the state s, the agent receives immediate reward r(s, a) according to the user's feedback

Transition $P$: $p(s'|s, a)$ defines the state transition from s to s' when RA takes action a

Discount factor $\gamma$: $\gamma \in [0, 1]$ when $\gamma = 0$, RA only considers the immediate reward; $\gamma = 1$, all future rewards can be counted fully into that of the current action.

In practice, conventional RL methods like Q-learning and POMDP become infeasible with the increasing number of items for recommendations. $\rightarrow$ leverage Deep Reinforcement Learning with (adapted) artificial neural networks as the non-linear approximators to estimate the action-value function in RL

Challenges: 

1. the large(or even continuous) and dynamic action space(item space)
2. the computational cost to select an optimal action 

Common ways:

​	use extra information to represent items with continuous embeddings

​	(the action space of recommender systems is dynamic && computing Q-value for all state-action pairs is time-consuming)

##### 2.1 Actor-Critic framework: 

suitable for large and dynamic action space; could also reduce redundant computation simultaneously compared to alternative architectures

Original optimal action-value function $Q^*(s, a)$ follows the Bellman equation:

​									$Q^*(s, a) = E_{s'}[r + \gamma max_{a'}Q^*(s', a')|s, a]$

In the Actor-Critic framework, the Actor architechture inputs the current state s and aims to output a deterministic action (or recommending a deterministic page of M items), the Critic inputs only this state-action pair (rather than all potential state-action pairs which avoids the aforementioned computational cost): 						$Q(s, a) = E_{s'}[r + \gamma Q(s', a')|s, a]$ 

$Q(s, a)$: a judgement of whether the selected action matches the current state(i.e., whether the recommendations match user's preference), then the Actor updates its' parameters in a direction of boosting recommendation performance.

##### 2.2 Architecture of Actor Framework

3 Challenges:

1. setting an initial preference at the beginning of a new recommendation session
2. learning the real-time preference in the current session(dynamic nature+item display patterns)
3. jointly generating a set of recommendations and displaying them in a 2-D page

*2.2.1 Encoder for Initial State Generation Process*

a RNN with GRU to capture users' sequential behaviors as user's initial preference;

Inputs of GRU: user's last clicked/purchased items $\{e_1,…,e_N\}$ (dense and low-dimensional, sorted in chronological order) before the current session

Outputs: the representation of users' initial preference by a vector

(?) Add an item-embedding layer to transform $e_i$ into a low-dimensional dense vector via $E_i = tanh(W_Ee_i + b_E) \in R^{|E|}$ where we use tanh activate function since $e_i \in (-1, +1)$.

*Why leverage GRU rather than LSTM*

unlike LSTM using input gate $i_t$ and forget gate $f_t$ to generate a new state, GRU utilizes an update gate $z_t$ :

​								$z_t = \sigma (W_zE_t + U_zh_{t-1})$

GRU leverages a reset gate $r_t$ to control the input of the formaer state $h_{t-1}$:

​								$r_t= \sigma (W_rE_t + U_rh_{t-1})$

The activation of GRU is a linear interpolation between the previous activation $h_{t-1}$ and the candidate activation $\hat{h_t}$:

​								$h_t = (1-z_t)h_{t-1}+ z_t\hat{h_t}$

​								$\hat{h_t} = tanh[WE_t+U(r_t\times h_{t-1})]$

The final hidden state $h_t$ as tje representation of the user's initial state $s^{ini}$ at the beginning of current recommendation session, i.e., $s^{ini} = h_t$

*2.2.2 Encoder for Real-time State Generation Process*

1. In page-wise recommender system, the inputs $\{x_1,…,x_M\}$ for each recommendation page are the representations of the items in the page and user's corresponding feedback, where M is the size of a recommendation page and $x_i$ is a tuple as 

​								$x_i = (e_i, c_i, f_i)$, 

$e_i$: item representation, $c_i$: item's category ($c_i(i) = 1$ if this item belongs to the $i^{th}$ category and other entities are zero) **one-hot indicator vector:** extremely sparse and high-dimensional.

 ——> add a category-embedding layer transforming $c_i$ into a low-dimensianl dense vector $C_i = tanh(W_Cc_i + b_C) \in R^{|C|}$.

$f_i$: user's feedbacl vector, one-hot vector to indicate user's feedback for item i.

——>transform $f_i$ into a dense vector $F_i = tanh(W_Ff_i+b_F) \in R^{|F|}$ via the embedding layer.

**A low-dimensional dense vector $X_i \in R^{|X|}$** ($|X| = |E| + |C| +|F|$)by concatenating $E_i, C_i, F_i$ as:

​								$X_i = concat(E_i, C_i, F_i)$

​									  $= tanh(concat(W_Ee_i+b_E, W_Cc_i+b_C,W_Ff_i+b_F))$ 

Since all item-embedding layers share the same parameters $W_E, b_E$ which reduces the number of parameters——>better generalization. (same constraints for category and feedback embedding layers)

2. **Reshape the transformed item representations**{$X_1, …, X_M$} as the original arrangement in the page.

=arrange the item representations in one page $P_t$ as 2D grids similar to one image.

eg: a recommendation page{h rows and w columns, M = $h\times w$}——> $h\times w|X|$ matrix $P_t$.

**CNN**: capability to apply various learnable kernel filters on image to discover complex spatial correlations; utilize 2D-CNN followed by fully connected layers to learn the optimal item display strategy as $p_t = conv2d(P_t)$, $p_t$ is a low-dimensional dense vector representing the information from the items and user's feedback in page $P_t$ && the spatial patterns of the item display strategy of page $P_t$.

3. Feed {$p_1, …, p_T$} into another RNN with GRU to capture user's real-time preference in the current session. Employ attention mechanism which allows the RA to adaptively focus on and linearly combine different parts of the input sequence: $s^{cur} = \sum_{t=1}^{T}\alpha_th_t, \alpha_t$ (weighted factor) determine which parts of the input sequence should be emphasized/ignored. ——>computed from the target hidden state $h_t$(leverage location-based attention mechanism): $\alpha_t = \frac{exp(W_{\alpha}h_t+b_{\alpha})}{\sum_jexp(W_{\alpha}h_j+b_{\alpha})}$.

a. the length of this GRU is flexible: i.e., user browse one page of generated recommendations and give feedback to RA ——> add one more GRU unit ——>use this page of items, corresponding categories and feedback as the input of the new GRU unit

b. 2 RNNs in 2.2.1 and 2.2.2 can be integrated into one RNN.

*2.2.3 Decoder for Action Generation Process* 

Deconvolution neural network(DeCNN) to restore a page from the low-dimensional representation $s^{cur}$:

​								$a^{cur} = deconv2d(s^{cur})$

a. $a^{cur}$ only contains item-embedding $E_i$, P also contains item's category embedding $C_i$ and feedback-embedding $F_i$;

——>i.e., a recommendation page: $M=h\times w, P=h\times w|X|, a^{cur}=h\times w|E|$.

b. the generated item embeddings in $a^{cur}$ may be not in the real item embedding set, thus we need to map them to valid item embeddings.

##### 2.3 The Architecture of Critic Framework

Critic: leverage an approximator to lean a Q(s, a)——> a judgement of whether the action a (or a recommendation page) generated by Actor matches the current state s.

Generate user's current state s: the RA follows the same strategy using embedding layers, 2D-CNN and GRU with attention mechanism. 

Since $a^{cur}$ generated above is a 2D matrix similar to an image, we utilize a 2D-CNN followed by fully connected layers to degrade $a^{cur}$ into a low-dimensional dense vector a: $a = conv2d(a^{cur})$.

Then the RA concatenates current state s and action a and feed them into a Q-value function Q(s, a). 

*Approximator: DQN(a neural network approximator as deep Q-value function)



