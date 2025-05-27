# AutoSentinel Phase 1: Development Environment

## Overview
AutoSentinel is an Autonomous Multi-Agent AI Cybersecurity Orchestrator that combines Vision Transformers (ViT), Multi-Agent Reinforcement Learning (MARL), and Large Language Models (LLMs) for real-time threat detection, analysis, and response.

## Directory Structure
```
autosentinel_system/
|-- data/
|   |-- raw/                    # Raw datasets (CICIDS2017)
|   |-- processed/              # Preprocessed and cleaned data
|   |-- visualizations/         # Network traffic visualizations for ViT
|   |-- marl_scenarios/         # MARL training scenarios
|-- models/
|   |-- vit/                    # Vision Transformer models
|   |-- marl/                   # Multi-Agent RL models
|   |-- trained/                # Saved model checkpoints
|-- configs/                    # Configuration files
|-- scripts/                    # Development and training scripts
|-- notebooks/                  # Jupyter notebooks for analysis
|-- tests/                      # Unit and integration tests
|-- docs/                       # Documentation
|-- logs/                       # Training and system logs
```

## Quick Start

### 1. Data Preprocessing
```bash
python scripts/preprocess_data.py
```
This script processes the CICIDS2017 dataset and creates train/validation/test splits.

### 2. Vision Transformer Preparation
```bash
python scripts/prepare_vit.py
```
Converts network flow data into 2D visualizations for ViT training.

### 3. MARL Environment Setup
```bash
python scripts/prepare_marl.py
```
Creates the multi-agent cybersecurity environment and training scenarios.

### 4. Run the Demo
```bash
python autosentinel_demo.py
```
Demonstrates the complete multi-agent system in action.

## Dataset Information

### CICIDS2017 Dataset
- **Source**: Canadian Institute for Cybersecurity
- **Features**: 78 network flow features
- **Classes**: 8 attack types
  - BENIGN
  - DDoS
  - PortScan  
  - Bot
  - Infiltration  - Web Attack - Brute Force
  - Web Attack - XSS
  - Web Attack - Sql Injection
- **Samples**: 10,000 synthetic samples for development

### Real Dataset Download
Download the complete CICIDS2017 dataset from:
https://www.unb.ca/cic/datasets/ids-2017.html

## Multi-Agent Architecture

### Agent Types
1. **ThreatDetectionAgent**: Uses ViT for pattern recognition in network traffic visualizations
2. **ResponseAgent**: Employs MARL for optimal response action selection
3. **AnalysisAgent**: Utilizes LLMs for threat analysis and report generation

### Coordination
Agents communicate through a centralized orchestrator that manages:
- Inter-agent communication
- Resource allocation
- Performance monitoring
- Incident logging

## Machine Learning Components

### Vision Transformer (ViT)
- **Purpose**: Convert network flows to visual patterns for threat detection
- **Input**: 16x16 grid representations of network flow features
- **Output**: Threat classification probabilities
- **Architecture**: 12-layer transformer with 768-dim embeddings

### Multi-Agent Reinforcement Learning (MARL)
- **Purpose**: Optimize response action coordination between agents
- **Environment**: Cybersecurity simulation with threat scenarios
- **Actions**: 6 response types (block, isolate, firewall, alert, quarantine, reset)
- **Rewards**: Based on threat mitigation and agent coordination

### Large Language Model Integration
- **Purpose**: Generate human-readable threat analysis reports
- **Input**: Threat events and response actions
- **Output**: Detailed incident reports with recommendations
- **Model**: Llama3-based cybersecurity analyst (simulated)

## Performance Metrics

### Detection Metrics
- Accuracy, Precision, Recall, F1-Score
- False Positive Rate
- Detection Time

### Response Metrics  
- Response Time
- Action Success Rate
- Threat Mitigation Effectiveness

### System Metrics
- Multi-agent Coordination Score
- Resource Utilization
- System Throughput

## Configuration

### Main Configuration (`configs/config.json`)
- Model parameters
- Training settings
- Data processing options

### Environment Configuration (`configs/environment.json`)
- Development/Staging/Production settings
- Logging levels
- Resource limits

### Dataset Configuration (`configs/dataset_config.json`)
- CICIDS2017 feature definitions
- Attack type mappings
- Data preprocessing parameters

## Development

### Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.11+
- scikit-learn 1.0+

### Installation
```bash
pip install -r requirements.txt
```

### Testing
```bash
python -m pytest tests/
```

## License
MIT License - see LICENSE file for details.

## Contributing
Please read CONTRIBUTING.md for guidelines on contributing to AutoSentinel.

## Support
For questions and support, please open an issue on the GitHub repository.
