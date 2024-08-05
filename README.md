In the environment, the agent is tasked with eating edible foods (+1 reward) and avoiding poisonous foods (-1 reward if eaten), represented as different coloured grid cells with colour denoting edibility (e.g., green for edible, red for poisonous). 
Additionally, the agents expend energy at each time step (reward of -0.01).

Agents can take actions up, down, left, right, or eat. They can see the colour (RGB value) of their current and adjacent grid cells.

The environment is subject to seasonal changes that alter the colour coding for food edibility. 
The number of seasons an agent goes through over their lifetime (100 timesteps) can be specified when initiating the environment (default: n_seasons=4). 
The environment only works for n_seasons values of 1, 2, 3, or 4.
