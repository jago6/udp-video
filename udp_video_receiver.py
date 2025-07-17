import cv2
import socket
import struct
import threading
import time
import logging
import numpy as np
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoReceiver:
    def __init__(self, host='127.0.0.1', port=12346, frame_timeout=1.0, max_frame_buffer=10):
        """
        Initialize video receiver
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            frame_timeout: Timeout for incomplete frames (seconds)
            max_frame_buffer: Maximum number of frames to buffer
        """
        self.host = host
        self.port = port
        self.frame_timeout = frame_timeout
        self.max_frame_buffer = max_frame_buffer
        
        # Create socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.settimeout(0.1)  # Non-blocking with timeout
        
        # Frame reconstruction buffers
        self.frame_buffers = defaultdict(dict)  # frame_id -> {packet_id: packet_data}
        self.frame_info = defaultdict(dict)     # frame_id -> {total_packets, received_packets, timestamp}
        self.completed_frames = deque()         # Queue of completed frames
        
        # Threading
        self.buffer_lock = threading.Lock()
        self.is_running = False
        
        # Statistics
        self.stats = {
            'packets_received': 0,
            'packets_out_of_order': 0,
            'frames_completed': 0,
            'frames_dropped': 0,
            'frames_displayed': 0
        }
        
        logger.info(f"Video receiver initialized on {host}:{port}")
    
    def parse_packet_header(self, packet):
        """Parse packet header to extract metadata"""
        if len(packet) < 24:  # Header size
            return None
        
        header = packet[:24]
        data = packet[24:]
        
        try:
            frame_id, packet_id, total_packets, data_size, timestamp = struct.unpack('!IIIIQ', header)
            return {
                'frame_id': frame_id,
                'packet_id': packet_id,
                'total_packets': total_packets,
                'data_size': data_size,
                'timestamp': timestamp,
                'data': data[:data_size]
            }
        except struct.error as e:
            logger.error(f"Error parsing packet header: {e}")
            return None
    
    def process_packet(self, packet):
        """Process incoming packet and update frame buffers"""
        packet_info = self.parse_packet_header(packet)
        if not packet_info:
            return
        
        frame_id = packet_info['frame_id']
        packet_id = packet_info['packet_id']
        total_packets = packet_info['total_packets']
        
        with self.buffer_lock:
            # Initialize frame info if not exists
            if frame_id not in self.frame_info:
                self.frame_info[frame_id] = {
                    'total_packets': total_packets,
                    'received_packets': 0,
                    'first_packet_time': time.time()
                }
            
            # Check if packet already received (duplicate)
            if packet_id in self.frame_buffers[frame_id]:
                logger.debug(f"Duplicate packet received: frame {frame_id}, packet {packet_id}")
                return
            
            # Store packet data
            self.frame_buffers[frame_id][packet_id] = packet_info['data']
            self.frame_info[frame_id]['received_packets'] += 1
            
            # Check if packet is out of order
            expected_next_packet = len(self.frame_buffers[frame_id]) - 1
            if packet_id != expected_next_packet:
                self.stats['packets_out_of_order'] += 1
                logger.debug(f"Out-of-order packet: frame {frame_id}, packet {packet_id}")
            
            self.stats['packets_received'] += 1
            
            # Check if frame is complete
            if self.frame_info[frame_id]['received_packets'] == total_packets:
                self.complete_frame(frame_id)
                logger.info(f"Frame {frame_id} completed ({total_packets} packets)")
    
    def complete_frame(self, frame_id):
        """Complete frame reconstruction and add to display queue"""
        try:
            # Reconstruct frame data by sorting packets
            frame_data = b''
            for packet_id in sorted(self.frame_buffers[frame_id].keys()):
                frame_data += self.frame_buffers[frame_id][packet_id]
            
            # Add to completed frames queue
            self.completed_frames.append({
                'frame_id': frame_id,
                'data': frame_data,
                'timestamp': time.time()
            })
            
            # Limit queue size
            while len(self.completed_frames) > self.max_frame_buffer:
                dropped_frame = self.completed_frames.popleft()
                self.stats['frames_dropped'] += 1
                logger.warning(f"Dropped frame {dropped_frame['frame_id']} due to buffer overflow")
            
            # Clean up buffers
            del self.frame_buffers[frame_id]
            del self.frame_info[frame_id]
            
            self.stats['frames_completed'] += 1
            
        except Exception as e:
            logger.error(f"Error completing frame {frame_id}: {e}")
    
    def cleanup_expired_frames(self):
        """Remove expired incomplete frames"""
        current_time = time.time()
        expired_frames = []
        
        with self.buffer_lock:
            for frame_id, info in self.frame_info.items():
                if current_time - info['first_packet_time'] > self.frame_timeout:
                    expired_frames.append(frame_id)
            
            for frame_id in expired_frames:
                received = self.frame_info[frame_id]['received_packets']
                total = self.frame_info[frame_id]['total_packets']
                logger.warning(f"Frame {frame_id} expired ({received}/{total} packets received)")
                
                del self.frame_buffers[frame_id]
                del self.frame_info[frame_id]
                self.stats['frames_dropped'] += 1
    
    def display_frame(self, frame_data):
        """Decode and display frame"""
        try:
            # Decode JPEG data
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Add statistics overlay
                stats_text = [
                    f"Packets: {self.stats['packets_received']}",
                    f"Frames: {self.stats['frames_completed']}",
                    f"Dropped: {self.stats['frames_dropped']}",
                    f"Out-of-order: {self.stats['packets_out_of_order']}"
                ]
                
                y_offset = 30
                for i, text in enumerate(stats_text):
                    cv2.putText(frame, text, (10, y_offset + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Receiver - Video Stream', frame)
                self.stats['frames_displayed'] += 1
                
                return True
            else:
                logger.error("Failed to decode frame")
                return False
                
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            return False
    
    def packet_receiver_thread(self):
        """Thread for receiving packets"""
        logger.info("Packet receiver thread started")
        
        while self.is_running:
            try:
                packet, addr = self.socket.recvfrom(2048)
                self.process_packet(packet)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error receiving packet: {e}")
    
    def frame_cleanup_thread(self):
        """Thread for cleaning up expired frames"""
        logger.info("Frame cleanup thread started")
        
        while self.is_running:
            self.cleanup_expired_frames()
            time.sleep(0.5)  # Check every 500ms
    
    def statistics_thread(self):
        """Thread for reporting statistics"""
        logger.info("Statistics thread started")
        
        while self.is_running:
            time.sleep(5)  # Report every 5 seconds
            logger.info(f"Receiver Stats - Packets: {self.stats['packets_received']}, "
                       f"Frames completed: {self.stats['frames_completed']}, "
                       f"Frames dropped: {self.stats['frames_dropped']}, "
                       f"Out-of-order: {self.stats['packets_out_of_order']}")
    
    def start_receiving(self):
        """Start video receiving"""
        self.is_running = True
        logger.info("Starting video reception...")
        
        # Start background threads
        threads = [
            threading.Thread(target=self.packet_receiver_thread, daemon=True),
            threading.Thread(target=self.frame_cleanup_thread, daemon=True),
            threading.Thread(target=self.statistics_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            # Main display loop
            while self.is_running:
                # Display completed frames
                if self.completed_frames:
                    with self.buffer_lock:
                        if self.completed_frames:
                            frame_info = self.completed_frames.popleft()
                            self.display_frame(frame_info['data'])
                
                # Check for exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received")
                    break
                
                time.sleep(1/60)  # 60 FPS display rate
                
        except KeyboardInterrupt:
            logger.info("Reception interrupted by user")
        finally:
            self.stop_receiving()
    
    def stop_receiving(self):
        """Stop video receiving and cleanup"""
        self.is_running = False
        cv2.destroyAllWindows()
        self.socket.close()
        
        # Final statistics
        logger.info("Video reception stopped")
        logger.info(f"Final Stats - Packets received: {self.stats['packets_received']}, "
                   f"Frames completed: {self.stats['frames_completed']}, "
                   f"Frames displayed: {self.stats['frames_displayed']}, "
                   f"Frames dropped: {self.stats['frames_dropped']}")

def main():
    """Main function to run the video receiver"""
    try:
        receiver = VideoReceiver(
            host='127.0.0.1',
            port=12346,
            frame_timeout=0.5,
            max_frame_buffer=10
        )
        receiver.start_receiving()
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
