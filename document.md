
# Part-Based Graph Convolutional Networks for Action Recognition

## Abstract

In this paper, we present a novel approach for action recognition using part-based graph convolutional networks (GCNs). We evaluate different architectures, including non-graph-based methods, part-based GCNs, spatio-temporal GCNs, and part-based graph convolutions. Our experiments demonstrate the effectiveness of our approach in achieving high accuracy on the UCF50 dataset.

## 1. Introduction

### 1.1 Background

Action recognition is a significant area of research in computer vision due to its wide range of applications, such as surveillance, human-computer interaction, video indexing, and sports analytics. The objective of action recognition is to identify and classify human actions in video sequences. Traditional methods for action recognition rely heavily on handcrafted features, such as the Histogram of Oriented Gradients (HOG), Optical Flow, and spatio-temporal interest points. These methods, however, have limitations in their ability to capture complex motion patterns and generalize across different datasets.

With the advent of deep learning, Convolutional Neural Networks (CNNs) have become the dominant approach for action recognition tasks. CNNs are capable of learning hierarchical feature representations from raw data, eliminating the need for handcrafted features. Despite their success, CNNs are limited in their ability to capture the spatial relationships between different parts of the human body, which are crucial for understanding actions.

### 1.2 Objective

Graph Convolutional Networks (GCNs) have emerged as a powerful tool for processing non-Euclidean data, making them suitable for capturing complex relationships in human actions. GCNs can model the spatial relationships between different body parts and their temporal evolution over time. In this paper, we explore the use of part-based GCNs for action recognition. Our objective is to develop models that can accurately classify actions by understanding the relationships between different body parts and their movements.

### 1.3 Contributions

Our contributions are as follows:

1. We introduce a part-based GCN architecture for action recognition that models the spatial relationships between different body parts.

2. We extend the part-based GCN to incorporate temporal information, resulting in a spatio-temporal GCN that captures the evolution of actions over time.

3. We propose a part-based graph convolutional network that combines multiple GCN layers to capture complex interactions and dependencies.

4. We provide a comprehensive evaluation of different architectures on the UCF50 dataset, demonstrating the effectiveness of our approach in achieving high accuracy.

## 2. Related Work

Action recognition has been extensively studied in the field of computer vision, with various approaches proposed over the years. In this section, we review the existing methods for action recognition, including traditional handcrafted feature-based methods, deep learning-based methods, and graph-based methods.

### 2.1 Handcrafted Features

Early approaches to action recognition relied on handcrafted features to represent the motion patterns in video sequences. Popular handcrafted features include the Histogram of Oriented Gradients (HOG), which captures gradient orientation information, and Optical Flow, which represents the motion of objects in consecutive frames. While these features were effective for simple actions, they struggled to capture complex motion patterns and were sensitive to variations in scale, viewpoint, and background clutter.

### 2.2 Deep Learning Models

With the advent of deep learning, Convolutional Neural Networks (CNNs) became the dominant approach for action recognition tasks. CNNs are capable of learning hierarchical feature representations from raw data, eliminating the need for handcrafted features. Models such as C3D (3D Convolutional Networks) and I3D (Inflated 3D Convolutional Networks) have shown significant improvements in action recognition performance. These models extend the 2D convolutions to 3D convolutions to capture spatio-temporal information from video sequences. However, CNNs still struggle to capture the spatial relationships between different body parts, which are crucial for understanding actions.

### 2.3 Graph-Based Methods

Graph-based methods have gained popularity in recent years due to their ability to model complex relationships in non-Euclidean data. In the context of action recognition, Graph Convolutional Networks (GCNs) have been used to model the relationships between different body parts. Yan et al. (2018) introduced the Spatial Temporal Graph Convolutional Networks (ST-GCN) which models human actions as a graph of joints. ST-GCN leverages the spatial and temporal relationships between joints to improve action recognition performance. However, ST-GCN models joints as nodes, which may not capture higher-level interactions between body parts.

### 2.4 Our Approach

Our approach differs from existing works by focusing on part-based graph convolutions, which explicitly model the relationships between different body parts. Unlike ST-GCN, which uses joints as nodes, our method uses larger body parts, providing a higher-level representation. This allows the model to capture more complex interactions and dependencies, leading to improved accuracy in recognizing actions.


Absolutely, I will now provide a detailed methodology section, explaining how our models utilize different body parts for action recognition. This will include the mathematical formulations, detailed architecture descriptions, and how the models process the data.
Absolutely, I'll include the formulas in LaTeX format in the methodology section and then continue with the Experimental Setup section.

---

## 3. Methodology

### 3.1 Overview
In this section, we describe the different architectures used in our study: Non-Graph-Based CNN, Part-Based Graph CNN, Spatio-Temporal Graph CNN, and Part-Based Graph Convolutions. Each model is designed to leverage the spatial and temporal relationships between different body parts to improve action recognition performance.

### 3.2 Non-Graph-Based CNN
The Non-Graph-Based CNN serves as our baseline model. It comprises standard convolutional and pooling layers followed by fully connected layers. This model does not explicitly model the relationships between different body parts but relies on the convolutional layers to learn the features from the input frames.

**Architecture:**
- **Input Layer:** Accepts frames resized to 128x128 pixels.
- **Convolutional Layer:** Applies convolution operations to extract features.
- **Max Pooling Layer:** Reduces the spatial dimensions.
- **Fully Connected Layer:** Maps the features to the output classes.

**Mathematical Formulation:**
\[
\text{Conv Layer:} \quad x' = \text{ReLU}(W * x + b)
\]
\[
\text{Pooling Layer:} \quad x'' = \text{MaxPool}(x')
\]
\[
\text{Fully Connected Layer:} \quad y = \text{Softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot x'' + b_1) + b_2)
\]

### 3.3 Part-Based Graph CNN
The Part-Based Graph CNN models the spatial relationships between different body parts. Each node in the graph represents a body part, and edges represent the connections between these parts. The graph structure allows the model to capture the dependencies and interactions between different body parts, leading to better feature representation.

**Architecture:**
- **Input Layer:** Accepts frames resized to 128x128 pixels.
- **Convolutional Layer:** Extracts features from the input frames.
- **Graph Convolutional Layer:** Models the relationships between body parts.
- **Fully Connected Layer:** Maps the features to the output classes.

**Mathematical Formulation:**
\[
h^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} W^{(l)} h_j^{(l)} + b^{(l)} \right)
\]

### 3.4 Spatio-Temporal Graph CNN
The Spatio-Temporal Graph CNN extends the Part-Based Graph CNN by incorporating temporal information. This model captures the evolution of actions over time by modeling the sequence of frames as a temporal graph. Each node represents a body part at a specific time step, and edges represent the spatial and temporal relationships between body parts.

**Architecture:**
- **Input Layer:** Accepts frames resized to 128x128 pixels.
- **Convolutional Layer:** Extracts features from the input frames.
- **Spatio-Temporal Graph Convolutional Layer:** Models the spatial and temporal relationships between body parts.
- **Fully Connected Layer:** Maps the features to the output classes.

**Mathematical Formulation:**
\[
h^{(l+1)} = \sigma \left( \sum_{t=1}^{T} \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} W^{(l)} h_{j,t}^{(l)} + b^{(l)} \right)
\]

### 3.5 Part-Based Graph Convolutions
The Part-Based Graph Convolutions model combines multiple GCN layers to capture complex interactions and dependencies between body parts. This model leverages the hierarchical structure of GCNs to learn multi-level representations of the spatial and temporal relationships.

**Architecture:**
- **Input Layer:** Accepts frames resized to 128x128 pixels.
- **Convolutional Layer:** Extracts features from the input frames.
- **Multiple Graph Convolutional Layers:** Captures complex relationships between body parts.
- **Fully Connected Layer:** Maps the features to the output classes.

**Mathematical Formulation:**
\[
h^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} W^{(l)} h_j^{(l)} + b^{(l)} \right)
\]
\[
h^{(L)} = \text{GCN}_{\text{layer}} \circ \cdots \circ \text{GCN}_{1}(h^{(0)})
\]

### 3.6 Architecture Diagram

Below is the architecture diagram illustrating the flow of data through the different components of the Part-Based Graph Convolutional Network for Action Recognition.

![Part-Based Graph Convolutional Network for Action Recognition](file-r5KGjB0oh1oD3NDSWbqqy7Gd)

The diagram shows how input frames are processed through convolutional layers, pooled, and then passed through graph convolutional layers that model the spatial and temporal relationships between different body parts, culminating in fully connected layers for final classification.

---

This completes the detailed methodology section with LaTeX-formatted formulas. Please review it, and if it looks good, we can proceed to the next section: Experimental Setup. If there are any changes needed, let me know, and we can adjust accordingly.