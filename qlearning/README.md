# Qlearning algorithm
## Exercise
The point of this exercise was to implement the qlearning algorithm
and train an agent to solve the taxi problem described [here](https://gymnasium.farama.org/environments/toy_text/taxi/).
I also impleneted and compared the effectiveness of Boltzmann and epsilon-greedy
exploration strategies.

## Results
In the end, both exploration strategies, as well as the decayed epsilon-greedy
strategy produced similar results.

#### Boltzmann strategy:
<p style="text-align:center;">
  <img src="plots/avg_rewards_boltzmann.png" width="500" />
  <img src="plots/episode_lenghts_boltzmann.png" width="500" />
</p>

#### Epsilon greedy strategy:
<p style="text-align:center;">
  <img src="plots/avg_rewards_epsilon.png" width="500" />
  <img src="plots/episode_lenghts_epsilon.png" width="500" />
</p>

## Used libraries
- numpy - numerical calculations
- gymnasium - environment operations
- matplotlib - plot creation
