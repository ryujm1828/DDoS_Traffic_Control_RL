# DDoS_Traffic_Control_RM

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

### Project Overview

This project is an implementation and extension of a reinforcement learning model for mitigating DDoS attacks, based on the concepts from the IEEE paper referenced below.

The core idea is to train a network router as an AI agent using a Deep Q-Network (DQN). This agent learns to autonomously decide when and how much to throttle incoming network traffic to protect a server from being overwhelmed.

To enhance realism beyond the original paper, this project features a more dynamic simulation environment with:
* **Diverse user profiles:** Including normal users, video streamers, and attackers, each with distinct traffic patterns.
* **Intermittent attack scenarios:** Attacks occur at random intervals, forcing the agent to learn to differentiate between peacetime and attack situations.

The trained agent successfully learned to apply throttling selectively only when an attack was detected, demonstrating the viability of using RL for intelligent, real-time DDoS defense.

### Related Link
 - [Simple Descriptions](https://ryusthought.blogspot.com/2025/06/implementing-paper-controlling-ddos.html)
 - [Description Video(To be added)]

### Reference

This work is based on the following paper:
> Shi-Ming Xia; Lei Zhang; Wei Bai; Xing-Yu Zhou; Zhi-Song Pan, "DDoS Traffic Control Using Transfer Learning DQN With Structure Information," in IEEE Access, vol. 7.
> 
> **Link:** [https://ieeexplore.ieee.org/document/8742593](https://ieeexplore.ieee.org/document/8742593)

### License
This project is licensed under the MIT License.
