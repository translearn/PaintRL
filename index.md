---
layout: default
---

----

<a name="toc"/>

<div style="text-align: center; font-size: 0.8em;">
<a href="#introduction">Introduction</a> &middot;
<a href="#motivation">Motivation</a> &middot;
<a href="#disclaimer">Disclaimer</a> &middot;
<a href="#data_generation">Data generation</a> &middot;
<a href="#simulation">Simulation</a> &middot;
<a href="#experiments">Experiments</a> &middot;
<a href="#results">Results</a> &middot;
<a href="#transfer">Sim2real transfer</a>
</div>

----

<a name="introduction"/>
## Introduction <a href="#toc" class="top-link">[Top]</a>

Spray painting is a widely used process for surface treatment especially in the metal industry. The processed parts obtain improved surface properties such as corrosion resistance and electric insulation. As a result of mass production and volatile organic solvents in the paint material, spray painting has been taken over by industrial robots for a long time.

PaintRL is a pure Python framework based on PyBullet, it supports the trajectory planning of industrial spray painting with reinforcement learning in simulation. Besides, classic planning algorithms such as [Andulkar et. al](https://linkinghub.elsevier.com/retrieve/pii/S0278612515000229) and [Chen and Xi](http://link.springer.com/10.1007/s00170-006-0746-5) is also compatible with the framework.

The paint can be further replaced by light via projection mapping to allow sim2real transfer (see <a href="#transfer">Sim2real transfer</a>), which opens up new possibilities to visualize the results from simulation, collect human demonstrations and capture real-world images. 

<a name="motivation"/>
## Motivation <a href="#toc" class="top-link">[Top]</a>

Industry 4.0 requires flexible line production. However, the teaching of the robot trajectory is a trial and error process which relies heavily on the experience of the domain experts. Taken the model below as analysis, trajectories for processing each part of the model should be programmed manually. 

<p align="center">
  <img src="assets/images/suzuki_anatomy.png"/>
</p>

In terms of spray painting, the main objective is to achieve uniform coating thickness with minimal material cost and shorter process time. Therefore many paint tools charge the paint material electrostatically and shape the form of the paint with air flow, which introduces several affections such as and gravity field, the speed received of the robot movement, external wind blows, uneven part surface profile. As a result, the trajectory planning for spray painting is still challenging.

The trajectory generation of spray painting has been studied over the last few decades. Beside the two publications mentioned above, the work from [Sheng et al.](http://ieeexplore.ieee.org/document/1458717/) is also widely adopted and cited. However, nowadays the surface of automobiles became more uneven, which makes the simplifications made in those publications less plausible. RL, especially DRL is a promising approach that is capable to solve problems under sophisticated constraints. It is therefore selected to explore the effective trajectory planning of spray painting.

<a name="data_generation"/>
## Data Generation <a href="#toc" class="top-link">[Top]</a>

We address the challenging task to collect training data for industrial tasks by
+ developing a fast and scalable spray painting simulation
+ replacing paint with light to collect real-world data

<a name="disclaimer"/>
## Disclaimer <a href="#toc" class="top-link">[Top]</a>

The primary target of this project is to perform trajectory planning and optimization for spray painting in a simulated environment. Since CAD mesh model is available for most industrial parts, it is taken as input of the planning algorithm, while the 6D coordinates of the planned trajectory is the output.

To use the planned trajectory in reality, the robot type could be selected according to the scale of the part and the tool. The reachability could be validated by calculating the inverse kinematics of each point on the trajectory.


<a name="simulation"/>
## Spray painting simulation <a href="#toc" class="top-link">[Top]</a>

+ The paint flux of a spray gun is modeled by a beta distribution

<p align="center">
  <img src="assets/images/beta_distribution.jpg" width="50%"/>
</p>

+ Impact points of paint droplets are calculated with ray-surface intersection tests provided by PyBullet

<p align="center">
  <img src="assets/images/paint_cone.png" width="50%"/>
</p>

+ The robot moves orthogonally to the surface normals of the workpiece

<p align="center">
  <img src="assets/images/paint_stroke.png"/>
</p>


<a name="experiments"/>
## Experiments <a href="#toc" class="top-link">[Top]</a>

The coverage path planning is formalized as a markov decision process (S, A, P<sub>a</sub>, R<sub>a</sub>)

### Observation:

+ Pose of the spray gun
+ Ratios of unpainted pixels and total pixels for circular sectors around the spray gun

<p align="center">
  <img src="assets/images/section_obs_door.png" width="50%"/>
</p>

### Actions:

+ Discrete actions which control the direction of the robot movement

<p align="center">
  <img src="assets/images/action_discrete.png" width="50%"/>
</p>

### Reward:

+ Number of newly painted pixels
+ Time penalty
+ Optional overlap penalty

### Baseline:

+ Quadratic sheet
+ Zigzag pattern

<p align="center">
  <img src="assets/images/zigzag_hsi.png" width="50%"/>
</p>

<a name="results"/>
## Results <a href="#toc" class="top-link">[Top]</a>

+ Generated path leads to full paint coverage of a car door
+ Time equivalent to baseline

**Figure of the results, video capture of the rollout**

<iframe width="853" height="480" src="https://www.youtube-nocookie.com/embed/TadnJeuAY6I?rel=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<a name="transfer"/>
## Sim2real transfer  <a href="#toc" class="top-link">[Top]</a>

Projection mapping opens up new possibilities to:

+ visualize the results from simulation
+ collect human demonstrations
+ capture real-world images

<div class="embed-container" align="center">
  <iframe width="853" height="480" src="https://www.youtube-nocookie.com/embed/nJVLpEk1MOs?rel=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
