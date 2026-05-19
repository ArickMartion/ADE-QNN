# Ancillary Dimensional Expansion Assisted Quantum Neural Network (ADE-QNN)

An ADE-QNN framework compatible with photonic quantum chips, designed for quantum machine learning tasks through data replication strategies and multi-value controlled RY (MCRY) gates.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MindQuantum](https://img.shields.io/badge/MindQuantum-0.11.0-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## 📋 Overview

This project implements a fully customizable Quantum Neural Network (QNN) architecture based on the MindQuantum framework, specifically designed for:

- **Photonic Quantum Chip Compatibility**  
  All modules are developed with consideration for synchronization and deployment on experimental photonic quantum hardware.

- **Quantum Machine Learning Applications**  
  Supporting a variety of quantum machine learning tasks, including classification problems.

- **Custom Gradient Control**  
  Manual implementation of forward propagation, gradient computation, and parameter optimization for enhanced flexibility and hardware adaptation.

- **Data Replication Strategy**  
  Employing ancillary-dimensional expansion through data replication to introduce stronger nonlinear capability.

- **Multi-value Controlled RY (MCRY) Gates**  
  Utilizing MCRY operations to realize effective non-unitary transformations within the QNN architecture.

---

## 🎯 Features

- **Custom Quantum Circuit Design**  
  Flexible construction of quantum circuits tailored for photonic quantum processors.

- **Manual Gradient Computation**  
  Full control over quantum gradient evaluation and parameter updates.

- **Data Replication Pipeline**  
  Quantum data encoding and replication mechanisms for nonlinear learning tasks.

- **Visualization Utilities**  
  Comprehensive plotting tools for quantum states, decision boundaries, and training progress.

- **Modular Architecture**  
  Clean separation of QNN components for convenient development and experimentation.

---

## 📁 Project Structure

```text
├── My_QNN.py                         # Custom QNN class definition
├── My_QCircuit.py                    # Quantum circuit construction
├── My_QClassification_Algorithm.py   # Classification algorithms and utilities
├── My_plot.py                        # Visualization and plotting functions

├── Circle_Classification.ipynb       # Example: circle classification task
├── Spiral_Classification.ipynb       # Example: spiral classification task

└── README.md                         # Project documentation
