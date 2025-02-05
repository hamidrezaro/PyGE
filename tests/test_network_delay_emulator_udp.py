import numpy as np
import matplotlib.pyplot as plt
from pyge.emulators.delay_emulator import DelayEmulator
import threading
import time
import socket
import struct
import lz4
import random
import string
import seaborn as sns
from collections import defaultdict
import json

from test_utils import ThreadWithReturn

def generate_random_payload(length=32):
    """Generate random string payload"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length)).encode()

def packet_sender(host: str, port: int, num_packets: int):
    """Send UDP packets with sequence numbers and timestamps"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[Sender] Starting to send {num_packets} packets...")
    start_time = time.time()
    
    for i in range(num_packets):
        timestamp = time.time_ns()
        payload = generate_random_payload()
        # Pack sequence number (I), timestamp (Q), and payload
        data = struct.pack(f'!IQ{len(payload)}s', i, timestamp, payload)
        sock.sendto(data, (host, port))
        
        if i % 5000 == 0 and i > 0:
            print(f"[Sender] Sent {i}/{num_packets} packets ({i/num_packets:.0%})")
    
    sock.close()
    duration = time.time() - start_time
    print(f"[Sender] Completed in {duration:.2f}s ({num_packets/duration:.0f} pkt/s)")

def packet_receiver(port: int, num_packets: int):
    """Receive UDP packets and track delays"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(5)  # Set receive timeout to 5 seconds
    
    delays = {}  # {sequence_number: (send_time, receive_time)}
    print(f"[Receiver] Listening on port {port} for {num_packets} packets...")
    start_time = time.time()
    last_print = 0
    last_received = 0
    timeout_counter = 0
    
    while len(delays) < num_packets and timeout_counter < 3:
        try:
            data, _ = sock.recvfrom(4096)
            receive_time = time.time_ns()
            
            # Unpack header (sequence number and timestamp)
            seq, send_time = struct.unpack('!IQ', data[:12])
            delays[seq] = (send_time, receive_time)
            
            last_received = time.time()
            timeout_counter = 0
            
            if time.time() - last_print > 2:
                print(f"[Receiver] Received {len(delays)}/{num_packets} packets ({len(delays)/num_packets:.0%})")
                last_print = time.time()
        
        except socket.timeout:
            if time.time() - last_received > 10:
                timeout_counter += 1
                print(f"[Receiver] Timeout {timeout_counter}/3 - No packets received for 10 seconds")
            continue
    
    sock.close()
    duration = time.time() - start_time
    print(f"[Receiver] Completed in {duration:.2f}s ({len(delays)/duration:.0f} pkt/s)")
    return delays

def analyze_delays(delays):
    """Analyze delay measurements and compute statistics"""
    print("[Analyzer] Computing delay statistics...")
    delay_values = []
    
    for seq in sorted(delays.keys()):
        send_time, recv_time = delays[seq]
        delay_ms = (recv_time - send_time) / 1_000_000  # Convert ns to ms
        delay_values.append(delay_ms)
    
    stats = {
        'total_packets': len(delays),
        'min_delay': np.min(delay_values),
        'max_delay': np.max(delay_values),
        'mean_delay': np.mean(delay_values),
        'median_delay': np.median(delay_values),
        'std_delay': np.std(delay_values),
        'p95_delay': np.percentile(delay_values, 95),
        'p99_delay': np.percentile(delay_values, 99),
        'delays': delay_values
    }
    return stats

def visualize_results(stats):
    """Create visualization of delay statistics"""
    print("[Visualization] Generating plots...")
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 0.5])
    
    # Delay Distribution Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=stats['delays'], bins=50, ax=ax1, 
                color='#2e86de', edgecolor='#1a3c6d')
    ax1.set_title('Delay Distribution', fontsize=14, pad=20)
    ax1.set_xlabel('Delay (ms)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # Delay Heatmap
    ax2 = fig.add_subplot(gs[:, 1])
    grid_width = 1000
    grid_height = 1000
    delay_grid = np.zeros((grid_height, grid_width))
    delays_normalized = np.array(stats['delays'])
    delays_normalized = (delays_normalized - stats['min_delay']) / (stats['max_delay'] - stats['min_delay'])
    
    print("[Visualization] Creating delay grid...")
    for i, delay in enumerate(delays_normalized):
        row = i % grid_height
        col = i // grid_height
        if col < grid_width:
            delay_grid[row][col] = delay
    
    print("[Visualization] Generating heatmap...")
    # Create heatmap with separate colorbar
    heatmap = sns.heatmap(delay_grid, cmap='viridis', ax=ax2,
                         xticklabels=False, yticklabels=False)
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Normalized Delay', fontsize=10)
    ax2.set_title('Packet Delay Pattern', fontsize=14, pad=20)
    
    # Statistics Panel
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis('off')
    
    stats_text = [
        "Network Delay Statistics",
        "─" * 25,
        f"Total Packets: {stats['total_packets']:,}",
        f"Min Delay: {stats['min_delay']:.2f} ms",
        f"Max Delay: {stats['max_delay']:.2f} ms",
        f"Mean Delay: {stats['mean_delay']:.2f} ms",
        f"Median Delay: {stats['median_delay']:.2f} ms",
        f"Std Dev: {stats['std_delay']:.2f} ms",
        f"95th Percentile: {stats['p95_delay']:.2f} ms",
        f"99th Percentile: {stats['p99_delay']:.2f} ms"
    ]
    
    ax3.text(0.1, 0.7, "\n".join(stats_text),
            fontsize=12, linespacing=1.8,
            color='#2d3436', fontfamily='monospace',
            bbox=dict(facecolor='#f8f9fa', edgecolor='#dfe6e9',
                    boxstyle='round,pad=1'))
    
    # Add CDF plot
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_delays = np.sort(stats['delays'])
    cumulative = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
    ax4.plot(sorted_delays, cumulative, color='#2e86de', linewidth=2)
    ax4.set_title('Cumulative Distribution Function', fontsize=14, pad=20)
    ax4.set_xlabel('Delay (ms)', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.grid(True)
    
    plt.suptitle("Network Delay Analysis",
                fontsize=18, y=0.98,
                color='#2d3436', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("[Visualization] Displaying results window")


if __name__ == "__main__":
    # Configuration
    EMULATOR_PORT = 5000
    RECEIVER_PORT = 5001
    NUM_PACKETS = 1_000_000  # Changed to 1 million packets
    
    # Load delay config
    print("[Main] Loading delay configuration...")
    with open('./src/pyge/canonical_configs/delay_config.json', 'r') as f:
        delay_params = json.load(f)

    # Start Network Emulator
    print("[Main] Starting network emulator...")
    delay_emulator = DelayEmulator(
        input_port=EMULATOR_PORT,
        output_port=RECEIVER_PORT,
        network_type='4G',
        params=delay_params,
        protocol='udp'
    )
    
    # Start receiver first using ThreadWithReturn
    print("[Main] Starting receiver thread...")
    receiver_thread = ThreadWithReturn(
        target=packet_receiver,
        args=(RECEIVER_PORT, NUM_PACKETS)
    )
    receiver_thread.start()
    
    # Start emulator
    print("[Main] Starting emulator...")
    delay_emulator.start()
    time.sleep(1)  # Wait for emulator to initialize
    
    # Start sender
    print("[Main] Starting sender thread...")
    sender_thread = threading.Thread(
        target=packet_sender,
        args=('localhost', EMULATOR_PORT, NUM_PACKETS)
    )
    sender_thread.start()
    
    # Wait for completion
    print("[Main] Test running...")
    sender_thread.join()
    print("[Main] Sender thread completed")
    
    delays = receiver_thread.join()  # Now this will properly return the delays dictionary
    print("[Main] Receiver thread completed")
    
    print("[Main] Stopping emulator...")
    delay_emulator.stop()
    print("[Main] Emulator stopped")
    
    # Analyze and visualize
    print("\n[Main] Analyzing results...")
    stats = analyze_delays(delays)
    print("\n[Main] Visualization:")
    visualize_results(stats) 