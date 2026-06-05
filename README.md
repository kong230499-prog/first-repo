# Delay-Coordinate State Reconstruction for Data-Driven Control

## Before running the project, make sure you have:

Python 3.9 or newer
NumPy
pandas
Matplotlib
SciPy
scikit-learn

Optional but recommended:

Jupyter Notebook or Google Colab for interactive testing
Git for version control

**Reconstructing full nonlinear system states from noisy partial measurements using delay-coordinate embeddings, regression models, and closed-loop feedback control.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Supported-lightgrey)
![SciPy](https://img.shields.io/badge/SciPy-Supported-lightgrey)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Supported-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## About / Overview

This project investigates whether **delay-coordinate state reconstruction** can recover missing state information from **noisy partial measurements** of nonlinear dynamical systems. The main test case observes only a noisy scalar measurement, such as position, while reconstructing the full state, including both position and velocity.

The project compares simple baselines against machine-learning models such as **Ridge regression** and a **multilayer perceptron neural network**. It also evaluates open-loop future prediction and a closed-loop feedback-control example using reconstructed states.

The project was developed for **ME 569** and focuses on:

- Simulating nonlinear dynamical systems
- Generating noisy partial measurements
- Building delay-coordinate embeddings
- Reconstructing full system states
- Comparing baseline, Ridge, and MLP models
- Testing open-loop future prediction
- Testing closed-loop feedback control using reconstructed states

---

## Key Features

- Simulates multiple nonlinear systems:
  - Van der Pol oscillator
  - Duffing oscillator
  - Nonlinear mass-spring-damper system

- Generates noisy partial measurements:
  - Example: noisy position-only measurement

- Constructs delay-coordinate vectors:

  ```text
  z_k = [y_k, y_{k-s}, y_{k-2s}, ..., y_{k-(d-1)s}]

## Project Structure

1. Imports
2. Output folders / portability setup
3. Nonlinear system definitions
4. Measurement and delay-coordinate functions
5. Baseline functions
6. Metrics
7. Model selection
8. Reconstruction experiment
9. Future-prediction experiment
10. Closed-loop control experiment
11. Plotting functions
12. Diagram functions
13. main()



  
