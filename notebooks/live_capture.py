# live_capture.py — Real-time Network Intrusion Detection

import pandas as pd
import numpy as np
import time
import joblib
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

# Load saved model, scaler and label encoder
print("Loading model...")
model = joblib.load('xgb_model.pkl')
scaler_obj = joblib.load('scaler.pkl')
le_obj = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')
print("Model loaded! Starting capture...\n")

# Flow tracker
flow_tracker = defaultdict(lambda: {
    'packets': 0, 'total_bytes': 0, 'start_time': None,
    'fwd_packets': 0, 'bwd_packets': 0,
    'fwd_bytes': 0, 'bwd_bytes': 0,
    'fin_count': 0, 'syn_count': 0, 'rst_count': 0,
    'psh_count': 0, 'ack_count': 0, 'urg_count': 0,
    'packet_lengths': [], 'dport': 0,
})

def get_flow_key(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        proto = packet[IP].proto
        sport = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0)
        dport = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
        return (src, dst, sport, dport, proto)
    return None

def extract_flow_features(flow):
    packets = flow['packets']
    duration = max(flow.get('duration', 1), 1)
    lengths = flow['packet_lengths'] if flow['packet_lengths'] else [0]

    features = {
        'Destination Port': flow.get('dport', 0),
        'Flow Duration': duration,
        'Total Fwd Packets': flow['fwd_packets'],
        'Total Backward Packets': flow['bwd_packets'],
        'Total Length of Fwd Packets': flow['fwd_bytes'],
        'Total Length of Bwd Packets': flow['bwd_bytes'],
        'Fwd Packet Length Max': max(lengths),
        'Fwd Packet Length Min': min(lengths),
        'Fwd Packet Length Mean': np.mean(lengths),
        'Fwd Packet Length Std': np.std(lengths),
        'Bwd Packet Length Max': max(lengths),
        'Bwd Packet Length Min': min(lengths),
        'Bwd Packet Length Mean': np.mean(lengths),
        'Bwd Packet Length Std': np.std(lengths),
        'Flow Bytes/s': flow['total_bytes'] / duration,
        'Flow Packets/s': packets / duration,
        'Flow IAT Mean': duration / max(packets, 1),
        'Flow IAT Std': 0, 'Flow IAT Max': duration, 'Flow IAT Min': 0,
        'Fwd IAT Total': duration,
        'Fwd IAT Mean': duration / max(flow['fwd_packets'], 1),
        'Fwd IAT Std': 0, 'Fwd IAT Max': duration, 'Fwd IAT Min': 0,
        'Bwd IAT Total': duration,
        'Bwd IAT Mean': duration / max(flow['bwd_packets'], 1),
        'Bwd IAT Std': 0, 'Bwd IAT Max': duration, 'Bwd IAT Min': 0,
        'Fwd PSH Flags': flow['psh_count'], 'Bwd PSH Flags': 0,
        'Fwd URG Flags': flow['urg_count'], 'Bwd URG Flags': 0,
        'Fwd Header Length': flow['fwd_packets'] * 20,
        'Bwd Header Length': flow['bwd_packets'] * 20,
        'Fwd Packets/s': flow['fwd_packets'] / duration,
        'Bwd Packets/s': flow['bwd_packets'] / duration,
        'Min Packet Length': min(lengths),
        'Max Packet Length': max(lengths),
        'Packet Length Mean': np.mean(lengths),
        'Packet Length Std': np.std(lengths),
        'Packet Length Variance': np.var(lengths),
        'FIN Flag Count': flow['fin_count'],
        'SYN Flag Count': flow['syn_count'],
        'RST Flag Count': flow['rst_count'],
        'PSH Flag Count': flow['psh_count'],
        'ACK Flag Count': flow['ack_count'],
        'URG Flag Count': flow['urg_count'],
        'CWE Flag Count': 0, 'ECE Flag Count': 0,
        'Down/Up Ratio': flow['bwd_packets'] / max(flow['fwd_packets'], 1),
        'Average Packet Size': flow['total_bytes'] / max(packets, 1),
        'Avg Fwd Segment Size': flow['fwd_bytes'] / max(flow['fwd_packets'], 1),
        'Avg Bwd Segment Size': flow['bwd_bytes'] / max(flow['bwd_packets'], 1),
        'Fwd Header Length.1': flow['fwd_packets'] * 20,
        'Fwd Avg Bytes/Bulk': 0, 'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0, 'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0, 'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': flow['fwd_packets'],
        'Subflow Fwd Bytes': flow['fwd_bytes'],
        'Subflow Bwd Packets': flow['bwd_packets'],
        'Subflow Bwd Bytes': flow['bwd_bytes'],
        'Init_Win_bytes_forward': 65535,
        'Init_Win_bytes_backward': 65535,
        'act_data_pkt_fwd': flow['fwd_packets'],
        'min_seg_size_forward': min(lengths),
        'Active Mean': 0, 'Active Std': 0, 'Active Max': 0, 'Active Min': 0,
        'Idle Mean': 0, 'Idle Std': 0, 'Idle Max': 0, 'Idle Min': 0,
    }
    return features

def process_packet(packet):
    key = get_flow_key(packet)
    if key is None:
        return

    flow = flow_tracker[key]

    if flow['start_time'] is None:
        flow['start_time'] = time.time()
        flow['dport'] = key[3]

    flow['duration'] = time.time() - flow['start_time']
    flow['packets'] += 1
    pkt_len = len(packet)
    flow['total_bytes'] += pkt_len
    flow['packet_lengths'].append(pkt_len)

    if packet[IP].src == key[0]:
        flow['fwd_packets'] += 1
        flow['fwd_bytes'] += pkt_len
    else:
        flow['bwd_packets'] += 1
        flow['bwd_bytes'] += pkt_len

    if TCP in packet:
        flags = packet[TCP].flags
        if flags & 0x01: flow['fin_count'] += 1
        if flags & 0x02: flow['syn_count'] += 1
        if flags & 0x04: flow['rst_count'] += 1
        if flags & 0x08: flow['psh_count'] += 1
        if flags & 0x10: flow['ack_count'] += 1
        if flags & 0x20: flow['urg_count'] += 1

    if flow['packets'] % 5== 0:
        features = extract_flow_features(flow)
        feature_df = pd.DataFrame([features])[feature_cols]
        scaled = scaler_obj.transform(feature_df)
        pred = model.predict(scaled)[0]
        label = le_obj.inverse_transform([pred])[0]

        timestamp = time.strftime('%H:%M:%S')
        if label != 'BENIGN':
            print(f"[{timestamp}] ⚠ ALERT: {label} detected! "
                  f"src={key[0]}:{key[2]} -> dst={key[1]}:{key[3]} "
                  f"| packets={flow['packets']}")
        else:
            print(f"[{timestamp}] OK  BENIGN | "
                  f"src={key[0]} -> dst={key[1]} "
                  f"| packets={flow['packets']}")

print("="*60)
print("  Network Intrusion Detection System - Live Capture")
print("="*60)
sniff(iface='\\Device\\NPF_{C13C3F2B-FBE4-413A-BBFB-43C23A65ABE5}',
      prn=process_packet, count=1000, store=False)
print("\nCapture complete!")