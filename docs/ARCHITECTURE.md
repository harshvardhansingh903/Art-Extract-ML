# Architecture Overview

## Project Architecture

This document describes the overall architecture of the Image ML Project.

### Task 1: Image Classification

**Purpose**: Multi-label image classification (Style, Artist, Genre)

**Architecture**:
- ResNet50 backbone with pre-trained weights
- Custom classification heads for each label type
- Data augmentation pipeline
- Cross-entropy loss training

### Task 2: Image Similarity

**Purpose**: Find similar images using learned embeddings

**Architecture**:
- ResNet50 backbone for feature extraction
- Embedding layer
- Cosine similarity search
- Optional contrastive/triplet loss training

### Shared Components

- Common data loading utilities
- Preprocessing and augmentation
- Metric computation helpers
- Device management (CPU/GPU)
