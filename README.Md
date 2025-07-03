# SNN Assignment Report

## Overview

Unlike Artificial Neural Networks (ANNs), which use continuous real-valued activations, SNNs communicate using discrete binary events (spikes) that occur over time. 



## Project Breakdown and Step-by-Step Explanation

### Step 1: Data Preprocessing and Spike Encoding


* Real biological neurons do not receive static images. Instead, they receive a stream of spikes that encode information. To simulate this, we used **Poisson rate coding** on the images.
* **Implementation**: For each pixel in the 28x28 image:

  * The pixel intensity (0 to 1) determines the probability of spiking at a given time step.
  * At every time step, for every pixel, we sample from a Bernoulli distribution.
  * Over `T` time steps (typically 100), this yields a binary spike train.

  * We also plot a heat map to visualize how the encoding works. The pixels having higher intensities(generally towards the center of the number image) almost always have a spike in the spike train, because of the encoding. The lesser intensity ones have a lower chance of having a spike. This simulates the input received by our brain


### Step 2: Network Architecture Design



We created a 3-layer SNN architecture:

* Input layer: 784 input neurons (one for each pixel)
* Hidden layer: 128 spiking neurons
* Output layer: 10 spiking neurons (corresponding to digit classes)



Each layer consists of two components:

* A linear transformation using `nn.Linear` to compute synaptic current.
* A custom **Leaky Integrate-and-Fire (LIF) neuron** that manages membrane potential, fires spikes, and resets.

We layered this in a `SNNLayer` class. The purpose of this abstraction was to mirror traditional feedforward neural networks, but with added temporal and spike-based behavior.

### Step 3: Leaky Integrate-and-Fire (LIF) Neuron Model



The LIF neuron is a simplified model of a biological neuron:

* The **membrane potential** integrates incoming input current.
* It **leaks** over time, meaning the potential decays gradually.
* When it exceeds a **threshold**, the neuron emits a spike.
* After spiking, the membrane potential **resets** to a lower value.

The equation governing this:

```
V(t+1) = V(t) + (I - V(t)) / tau
```

This equation balances the input current and the leak. This equation ensures that the activation is dynamic in nature, which evolves over time. In ANNs, this remains constant throughout. 

### Step 4: Surrogate Gradient Learning

We use the **surrogate gradient** method because it's better in classification problems, as then we can use the labels of images and calculate losses. On the other hand, using **STDP** would have made the process of learning unsupervised, where the we cannot the model won't learn using the labels directly. Hence, it's better to use surrogate gradient method for our classification task.
Spikes are non-differentiable (binary), so standard backpropagation breaks. We use the **surrogate gradient** method to deal with this problem.

* **Forward Pass**: Still emits binary spikes using a threshold.
* **Backward Pass**: Replaces the derivative of the step function with a smooth approximation (e.g. fast sigmoid).

The class `SurrogateSpike` handles this dual behavior, allowing gradients to flow even through binary events. This enables supervised learning in spiking networks.

### Step 5: Model Forward Pass

Unlike ANNs, where all computation happens in one pass, SNNs operate over time:

* At each time step:

  * A new input spike is fed into the network.
  * LIF neurons integrate the input, emit spikes, and reset.
* We **accumulate spikes** at the output layer across time.

This cumulative count becomes our model output and reflects class confidence. So if a neuron has more number of spikes after some time, that means that neuron is firing more confidently. We have 10 output neurons in our model, so if say a neuron corresponding to the value 7 is firing, then it's likely that our model believes the input to be 7.

### Step 6: Loss Function and Optimization

Since we're using supervised learning, we need to assign numerical targets. Although spike counts are not traditional logits, treating them as such allows us to:

* Apply `CrossEntropyLoss`
* Compare spike activity with class labels

The optimizer (Adam) updates the network weights using gradients provided by the surrogate gradient.

### Step 7: Training Loop

#### What We Did

We trained the model over multiple epochs while resetting neuron states between batches.

Key parts of the training loop:

* Reset LIF neurons before every forward pass.
* Feed spike-encoded inputs for T time steps.
* Accumulate output spikes.
* Compute loss and backpropagate.
* Track training and test accuracy.

Evaluation is done after each epoch to monitor progress.

---

## Evaluation and Analysis

### Metrics

* Classification accuracy on test set
* Spike activity and sparsity

### Key Observations

* **Surrogate gradients work**: Even though it seems very counter-intuitve, but training is possible despite only having binary spikes. The training is far more efficient as compared to ANNs in terms of the number of epochs, but also takes a lot more time.


### Advantages of SNNs

* Energy-efficient and event-driven computation
* Temporal encoding of input information
* Biologically plausible learning mechanisms



---

## Running the Project

### Requirements

```bash
pip install torch torchvision matplotlib
```

### Run Training

```bash
python train_snn.py
```


---


