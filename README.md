# Development and Comparative Analysis of MPC and PID Controllers for Autonomous Vehicles

## Overview

This project focuses on the development and comparison of two control strategies — Model Predictive Control (MPC) and Proportional-Integral-Derivative (PID) — for trajectory tracking in autonomous vehicles. The MPC algorithm is known for its predictive capabilities and optimization, while the PID is a classical control method with simpler implementation but less adaptability in complex environments.

The simulation was implemented entirely in Python and compares how both controllers perform in terms of trajectory accuracy, control smoothness, and computational efficiency. The vehicle dynamics model used is realistic and takes into account non-linear behavior, which enables meaningful results when testing the controllers.

---

## Technologies Used

- **Python** – Core implementation language
- **Matplotlib** – For data visualization and real-time animations
- **NumPy** – Scientific computing and matrix operations
- **cvxopt** – Convex optimization library used to solve the MPC quadratic programming problem
- **GitLab** – Version control and collaboration
- **Visual Studio Code** – Main development environment

---

## Project Structure
- MPC.py # Main file for Model Predictive Control 
- PID.py # Main file for PID controller 
- support_files_car.py # Physical model and utility functions 
- README.md # Project documentation (this file) 
- .venv/ # Python virtual environment (not pushed to GitLab) 
- requirements.txt #

---

## How to Run the Project

1. Clone the repository:
    git clone https://gitlab.com/mpc_pid_controller/mpc
    cd mpc

2. Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install the dependencies:
    pip install -r requirements.txt

4. Run the simulation:
    python MPC.py

---

## Author
Eduardo de Quiroga Tello
Student ID: 19314173
Oxford Brookes University

---

## Video Demonstration
Google Drive link: https://drive.google.com/file/d/1vSvt96XEjO6DG1ZAwPiL-J2xPlodLKcc/view?usp=drive_link  
