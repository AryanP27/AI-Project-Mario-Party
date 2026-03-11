# AIHPTermProject

I worked on this project for my AI class. It was the final project and my friend and I wanted to build something fun while also having the chance to do some statistical analysis. 

We chose three minigames from Mario Party. The game I primarily focused on was WhompStomp from Mario Party 9. 
The goal of this project was to create a neural network that used Q-learning to learn its environment and how rewards in the game work. 

We got some probabilty charts for the games from Youtuber ZoomZike, but watch this video to understand the WhompStomp game better at time 5:07:10: https://www.youtube.com/watch?v=2kc5X8Yt3oM 

We had the agent play as player 1 and the other three players did a specific policy (picking 1, 0, and picking randomly). 
To decide the agent's strategy, we used exploration vs exploitation. We set the Epsilon to 0.1 which means we use what we learned 90% of the time and only chose randomly 10 percent of the time. 

After building the rest of the neural network we tested it. For clarity purposes, the workflow looks like this: Game.py  & Train.py  → Test.py  → Data.py 

Game.py recreates the actual game while Train.py creates the neural network and it's environment. Test.py runs the game n times and gathers data which it then sends to Data.py which lets us visualize the data.

With that out of the way, we saw that after training our agent for 500 episodes, it had winrates of around 30-40%. This shows that our agent works properly since the expected winrate is 25%. 

The winrate could potentially be higher if we trained for more episodes, increased the network size, and adjusted a few parameters.

Overall this was one of my most favorite projects and I enjoyed it. 
