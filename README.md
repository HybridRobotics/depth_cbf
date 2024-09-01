# Point Cloud-Based Control Barrier Function Regression for Safe and Efficient Vision-Based Control

#### Abstract:
Control barrier functions have become an increasingly popular framework for safe real-time control. In this work, we present a computationally low-cost framework for synthesizing barrier functions over pointcloud data for safe vision-based control. We take advantage of surface geometry to locally define and fit a quadratic CBF over a pointcloud. This CBF is synthesized in a CBF-QP for control and verified in simulation on quadrotors and in hardware on quadrotors and the TurtleBot3. This technique enables safe navigation through unstructured and dynamically changing environments and is shown to be significantly more efficient than current methods.


#### Repository:
Simulation examples for the quadrotor and a hardware implementation example are provided in the package. Note that to run the quadrotor simulations, one should pip install the library CalSim.

