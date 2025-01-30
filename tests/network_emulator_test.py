import numpy as np
import matplotlib.pyplot as plt
from pyge.network_emulator import NetworkEmulator
import threading
import time
import socket
import struct
import lz4

def packet_sender(host: str, port: int, num_packets: int):
    """Send UDP packets with sequence numbers"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[Sender] Starting to send {num_packets} packets...")
    start_time = time.time()
    
    for i in range(num_packets):
        data = struct.pack('!I', i)  # 4-byte sequence number
        sock.sendto(data, (host, port))
        if i % 1000 == 0 and i > 0:
            print(f"[Sender] Sent {i}/{num_packets} packets ({i/num_packets:.0%})")
        time.sleep(0.0001)  # Small delay to avoid overwhelming
    
    sock.close()
    duration = time.time() - start_time
    print(f"[Sender] Completed in {duration:.2f}s ({num_packets/duration:.0f} pkt/s)")

def packet_receiver(port: int, num_packets: int):
    """Receive UDP packets and track received sequence numbers"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(5)  # Set receive timeout to 5 seconds
    received = set()
    print(f"[Receiver] Listening on port {port} for {num_packets} packets...")
    start_time = time.time()
    last_print = 0
    last_received = 0
    timeout_counter = 0
    
    while len(received) < num_packets and timeout_counter < 1:
        try:
            data, _ = sock.recvfrom(4096)
            seq = struct.unpack('!I', data)[0]
            received.add(seq)
            last_received = time.time()
            timeout_counter = 0  # Reset counter on successful receive
            
            # Print progress every 2 seconds
            if time.time() - last_print > 2:
                print(f"[Receiver] Received {len(received)}/{num_packets} packets ({len(received)/num_packets:.0%})")
                last_print = time.time()
        
        except socket.timeout:
            # Check if we've had consecutive timeouts
            if time.time() - last_received > 10:
                timeout_counter += 1
                print(f"[Receiver] Timeout {timeout_counter}/3 - No packets received for 10 seconds")
            continue
    
    sock.close()
    duration = time.time() - start_time
    
    if timeout_counter >= 3:
        print("[Receiver] ERROR: Aborted due to consecutive timeouts")
    elif len(received) < num_packets:
        print(f"[Receiver] WARNING: Only received {len(received)}/{num_packets} packets")
    
    print(f"[Receiver] Completed in {duration:.2f}s ({len(received)/duration:.0f} pkt/s)")
    return received

def analyze_log(log_file: str):
    """Analyze packet log and compute statistics"""
    print(f"[Analyzer] Starting analysis of {log_file}...")
    start_time = time.time()
    lost_packets = []
    current_burst = 0
    burst_lengths = []
    total_packets = 0
    
    with lz4.frame.open(log_file, 'rb') as f:
        while True:
            header = f.read(11)  # Q(8) + H(2) + ?(1)
            if not header:
                break
            ts, length, dropped = struct.unpack("!QH?", header)
            data = struct.unpack(f"!{length}s", f.read(length))[0]
            
            seq = struct.unpack('!I', data[:4])[0]
            total_packets = max(total_packets, seq + 1)
            if dropped:
                lost_packets.append(seq)
                current_burst += 1
            elif current_burst > 0:
                burst_lengths.append(current_burst)
                current_burst = 0
    
    # Final burst if log ends with loss
    if current_burst > 0:
        burst_lengths.append(current_burst)
    duration = time.time() - start_time
    print(f"[Analyzer] Completed in {duration:.2f}s ({total_packets/duration:.0f} pkt/s)")
    return {
        'total_packets': total_packets,
        'lost_packets': lost_packets,
        'loss_rate': len(lost_packets) / total_packets,
        'burst_lengths': burst_lengths,
        'total_bursts': len(burst_lengths)
    }

def visualize_results(stats: dict):
    """Create visualization of packet loss statistics"""
    print("[Visualization] Generating plots...")
    plt.style.use('ggplot')  # Use matplotlib's built-in ggplot style
    fig = plt.figure(figsize=(18, 8))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.4])
    
    # Burst Length Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.hist(stats['burst_lengths'], 
                   bins=range(1, max(stats['burst_lengths'])+2), 
                   align='left',
                   color='#2e86de',
                   edgecolor='#1a3c6d',
                   linewidth=1.2)
    
    # Add count labels on top of bars
    for rect in bars[2]:
        height = rect.get_height()
        if height > 0:
            ax1.text(rect.get_x() + rect.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_title('Burst Length Distribution', fontsize=14, pad=20, color='#2d3436')
    ax1.set_xlabel('Burst Length (consecutive lost packets)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Packet Grid Visualization
    ax2 = fig.add_subplot(gs[0, 1])
    grid_size = 100
    packet_grid = np.zeros((grid_size, grid_size))
    
    lost_set = set(stats['lost_packets'])
    for seq in range(stats['total_packets']):
        i, j = divmod(seq, grid_size)
        if seq in lost_set:
            packet_grid[i][j] = 1

    cmap = plt.cm.colors.ListedColormap(['#00b894', '#d63031'])
    ax2.imshow(packet_grid, cmap=cmap, interpolation='none', aspect='auto')
    ax2.set_title('Packet Loss Pattern', fontsize=14, pad=20, color='#2d3436')
    ax2.axis('off')
    
    # Create legend for grid
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='#00b894', markersize=10,
                                label='Delivered'),
                      plt.Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='#d63031', markersize=10,
                                label='Lost')]
    ax2.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
    
    # Statistics Panel
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis('off')
    
    stats_text = [
        "Network Loss Statistics",
        "\n",
        f"Total Packets: {stats['total_packets']}",
        f"Loss Rate: {stats['loss_rate']:.2%}",
        f"Total Bursts: {stats['total_bursts']}",
        f"Average Burst Length: {np.mean(stats['burst_lengths']):.2f}",
        f"Max Burst Length: {np.max(stats['burst_lengths']) if stats['burst_lengths'] else 0}"
    ]
    
    ax3.text(0.1, 0.7, "\n".join(stats_text), 
           fontsize=12, linespacing=1.8, 
           color='#2d3436', fontfamily='monospace',
           bbox=dict(facecolor='#f8f9fa', edgecolor='#dfe6e9', 
                   boxstyle='round,pad=1'))
    
    # Add summary title
    plt.suptitle("Network Packet Loss Analysis", 
               fontsize=18, y=0.98, 
               color='#2d3436', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("[Visualization] Displaying results window")

if __name__ == "__main__":
    # Configuration
    EMULATOR_PORT = 5000
    RECEIVER_PORT = 5001
    NUM_PACKETS = 10000
    LOG_FILE = "network_test_log.bin"
    
    # Start Network Emulator
    print("[Main] Starting network emulator...")
    emulator = NetworkEmulator(
        input_port=EMULATOR_PORT,
        output_port=RECEIVER_PORT,
        model_name='GEClassic',
        params_path='./src/pyge/config.json',
        protocol='udp',
        log_packets=True,
        log_path=LOG_FILE
    )
    
    # Start receiver first
    print("[Main] Starting receiver thread...")
    receiver_thread = threading.Thread(
        target=packet_receiver, 
        args=(RECEIVER_PORT, NUM_PACKETS)
    )
    receiver_thread.start()
    
    # Start emulator
    print("[Main] Starting emulator...")
    emulator.start()
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
    receiver_thread.join()
    print("[Main] Receiver thread completed")
    
    print("[Main] Stopping emulator...")
    emulator.stop()
    print("[Main] Emulator stopped")
    
    # Analyze and visualize
    print("\n[Main] Analyzing results...")
    stats = analyze_log(LOG_FILE)
    print("\n[Main] Visualization:")
    visualize_results(stats) 