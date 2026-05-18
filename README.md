# Network Intrusion Detection System ??

A real-time network intrusion detection system that classifies network traffic into 12 attack categories using machine learning, with live packet capture and an interactive Streamlit dashboard.

## Results
| Model | Accuracy |
|---|---|
| Random Forest | 98.76% |
| XGBoost | 98.81% |

## Attack Types Detected
| Attack | F1 Score |
|---|---|
| DDoS | 1.00 |
| DoS Hulk | 1.00 |
| DoS GoldenEye | 1.00 |
| PortScan | 1.00 |
| FTP-Patator | 1.00 |
| SSH-Patator | 1.00 |
| Bot | 0.99 |
| DoS Slowhttptest | 0.99 |
| DoS slowloris | 0.99 |
| Web Attack Brute Force | 0.73 |

## Project Structure
## Features
- Trained on 2.8 million real network flow records (CICIDS 2017)
- Handles severe class imbalance using SMOTE oversampling
- Multi-class classification across 12 attack types
- Real-time live packet capture and classification using Scapy
- Interactive Streamlit dashboard for visualization

## Tech Stack
Python, XGBoost, Scikit-learn, Scapy, Streamlit, Pandas, SMOTE

## How to Run

### Install dependencies
### Train the model
### Launch the Streamlit dashboard
### Run live capture (requires admin)
## Dataset
CICIDS 2017 — Canadian Institute for Cybersecurity
https://www.unb.ca/cic/datasets/ids-2017.html
