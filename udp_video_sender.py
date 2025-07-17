import cv2
import socket
import struct
import threading
import time
import logging
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoSender:
    def __init__(self, host='127.0.0.1', port=12345, max_packet_size=1024):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frame_sequence = 0
        self.packet_sequence = 0
        self.is_running = False
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Camera not available")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"Video sender initialized on {host}:{port}")
    
    def create_packet_header(self, frame_id, packet_id, total_packets, data_size):
        """
        Create packet header with sequence information
        Header format: frame_id(4) + packet_id(4) + total_packets(4) + data_size(4) + timestamp(8)
        """
        timestamp = int(time.time() * 1000000)  # microseconds
        header = struct.pack('!IIIIQ', frame_id, packet_id, total_packets, data_size, timestamp)
        return header
    
    def split_frame_to_packets(self, frame_data, frame_id):
        """Split frame data into multiple UDP packets"""
        packets = []
        data_per_packet = self.max_packet_size - 24  # Reserve space for header
        total_packets = (len(frame_data) + data_per_packet - 1) // data_per_packet
        
        for i in range(total_packets):
            start_idx = i * data_per_packet
            end_idx = min((i + 1) * data_per_packet, len(frame_data))
            packet_data = frame_data[start_idx:end_idx]
            
            header = self.create_packet_header(frame_id, i, total_packets, len(packet_data))
            packet = header + packet_data
            packets.append(packet)
        
        logger.debug(f"Frame {frame_id} split into {total_packets} packets")
        return packets
    
    def send_frame(self, frame):
        """Encode and send a single frame"""
        try:
            # Encode frame to JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_data = encoded_frame.tobytes()
            
            # Split frame into packets
            packets = self.split_frame_to_packets(frame_data, self.frame_sequence)
            
            # Send all packets for this frame
            for packet in packets:
                self.socket.sendto(packet, (self.host, self.port))
                self.packet_sequence += 1
            
            logger.info(f"Sent frame {self.frame_sequence} ({len(packets)} packets, {len(frame_data)} bytes)")
            self.frame_sequence += 1
            
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
    
    def start_streaming(self):
        """Start video streaming"""
        self.is_running = True
        logger.info("Starting video streaming...")
        
        try:
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Display local video (optional)
                cv2.imshow('Sender - Local Video', frame)
                
                # Send frame
                self.send_frame(frame)
                
                # Control frame rate (30 FPS)
                time.sleep(1/30)
                
                # Check for exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        finally:
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop video streaming and cleanup"""
        self.is_running = False
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()
        self.socket.close()
        logger.info("Video streaming stopped")

def main():
    """Main function to run the video sender"""
    try:
        sender = VideoSender(host='127.0.0.1', port=12345, max_packet_size=1024)
        sender.start_streaming()
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
