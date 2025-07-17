import cv2
import socket
import struct
import threading
import time
import logging
import numpy as np
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoReceiver:
    # ***** 新增了 video_width, video_height, video_channels 参数 *****
    def __init__(self, host='127.0.0.1', port=12346, frame_timeout=1.0, max_frame_buffer=10, 
                 video_width=640, video_height=480, video_channels=3):
        """
        初始化视频接收器

        Args:
            host: 绑定的主机地址
            port: 监听的端口
            frame_timeout: 不完整帧的超时时间 (秒)
            max_frame_buffer: 最大缓冲的帧数
            video_width: 视频帧的宽度
            video_height: 视频帧的高度
            video_channels: 视频帧的通道数 (例如 BGR 为 3)
        """
        self.host = host
        self.port = port
        self.frame_timeout = frame_timeout
        self.max_frame_buffer = max_frame_buffer
        
        # ***** 存储视频尺寸信息 *****
        self.video_width = video_width
        self.video_height = video_height
        self.video_channels = video_channels
        self.expected_frame_size = self.video_width * self.video_height * self.video_channels
        
        # 创建套接字
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        buffer_size = 2 * 1024 * 1024  # 2MB
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        logger.info(f"Socket 接收缓冲区大小已设置为 {buffer_size / 1024 / 1024} MB")
        self.socket.bind((host, port))
        self.socket.settimeout(0.1)  # 非阻塞并带超时

        # 帧重组缓冲区
        self.frame_buffers = defaultdict(dict)  # frame_id -> {packet_id: packet_data}
        self.frame_info = defaultdict(dict)     # frame_id -> {total_packets, received_packets, timestamp}
        self.completed_frames = deque()         # 已完成帧的队列

        # 线程
        self.buffer_lock = threading.Lock()
        self.is_running = False

        # 统计信息
        self.stats = {
            'packets_received': 0,
            'packets_out_of_order': 0,
            'frames_completed': 0,
            'frames_dropped': 0,
            'frames_displayed': 0
        }

        logger.info(f"视频接收器已在 {host}:{port} 初始化")
        logger.info(f"等待 {video_width}x{video_height}x{video_channels} 的原始视频流")

    def parse_packet_header(self, packet):
        """解析数据包头部以提取元数据"""
        header_size = 16  # 头部大小 (字节)
        if len(packet) < header_size:
            return None
        
        header = packet[:header_size]
        data = packet[header_size:]

        try:
            frame_id, packet_id, total_packets, data_size = struct.unpack('!IIII', header)
            return {
                'frame_id': frame_id,
                'packet_id': packet_id,
                'total_packets': total_packets,
                'data_size': data_size,
                'data': data[:data_size]
            }
        except struct.error as e:
            logger.error(f"解析数据包头部时出错: {e}")
            return None

    def process_packet(self, packet):
        """处理传入的数据包并更新帧缓冲区"""
        packet_info = self.parse_packet_header(packet)
        if not packet_info:
            return

        frame_id = packet_info['frame_id']
        packet_id = packet_info['packet_id']
        total_packets = packet_info['total_packets']

        with self.buffer_lock:
            if frame_id not in self.frame_info:
                self.frame_info[frame_id] = {
                    'total_packets': total_packets,
                    'received_packets': 0,
                    'first_packet_time': time.time()
                }
            
            if packet_id in self.frame_buffers[frame_id]:
                logger.debug(f"收到重复数据包: 帧 {frame_id}, 包 {packet_id}")
                return

            self.frame_buffers[frame_id][packet_id] = packet_info['data']
            self.frame_info[frame_id]['received_packets'] += 1
            
            expected_next_packet = len(self.frame_buffers[frame_id]) - 1
            if packet_id != expected_next_packet:
                self.stats['packets_out_of_order'] += 1
                logger.debug(f"乱序数据包: 帧 {frame_id}, 包 {packet_id}")
            
            self.stats['packets_received'] += 1
            
            if self.frame_info[frame_id]['received_packets'] == total_packets:
                self.complete_frame(frame_id)
                logger.info(f"帧 {frame_id} 已完成 ({total_packets} 个包)")

    def complete_frame(self, frame_id):
        """完成帧重组并添加到显示队列"""
        try:
            # 通过对数据包排序来重构帧数据
            frame_data_list = []
            for packet_id in sorted(self.frame_buffers[frame_id].keys()):
                frame_data_list.append(self.frame_buffers[frame_id][packet_id])
            frame_data = b''.join(frame_data_list)
            
            self.completed_frames.append({
                'frame_id': frame_id,
                'data': frame_data,
                'timestamp': time.time()
            })

            while len(self.completed_frames) > self.max_frame_buffer:
                dropped_frame = self.completed_frames.popleft()
                self.stats['frames_dropped'] += 1
                logger.warning(f"由于缓冲区溢出，丢弃帧 {dropped_frame['frame_id']}")

            del self.frame_buffers[frame_id]
            del self.frame_info[frame_id]
            
            self.stats['frames_completed'] += 1
            
        except Exception as e:
            logger.error(f"完成帧 {frame_id} 时出错: {e}")

    def cleanup_expired_frames(self):
        """移除过期的不完整帧"""
        current_time = time.time()
        expired_frames = []
        
        with self.buffer_lock:
            # 使用 list(self.frame_info.items()) 来避免在迭代时修改字典
            for frame_id, info in list(self.frame_info.items()):
                if current_time - info['first_packet_time'] > self.frame_timeout:
                    expired_frames.append(frame_id)
            
            for frame_id in expired_frames:
                received = self.frame_info[frame_id]['received_packets']
                total = self.frame_info[frame_id]['total_packets']
                logger.warning(f"帧 {frame_id} 已过期 ({received}/{total} 个包)")
                
                del self.frame_buffers[frame_id]
                del self.frame_info[frame_id]
                self.stats['frames_dropped'] += 1

    # ***** 这是主要修改的方法 *****
    def display_frame(self, frame_data):
        """解析原始数据并显示帧"""
        try:
            # 检查接收到的数据大小是否与预期的帧大小匹配
            if len(frame_data) != self.expected_frame_size:
                logger.error(f"帧数据大小不匹配! 预期: {self.expected_frame_size}, 收到: {len(frame_data)}")
                return False

            # 将字节数据转换为 NumPy 数组
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            
            # 将一维数组重塑为 (height, width, channels) 的图像格式
            # OpenCV 使用 BGR 顺序
            frame = frame.reshape((self.video_height, self.video_width, self.video_channels))
            frame = frame.copy()
            
            if frame is not None:
                # 添加统计信息覆盖层
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
                
                # 显示帧
                cv2.imshow('Receiver - Video Stream', frame)
                self.stats['frames_displayed'] += 1
                
                return True
            else:
                logger.error("无法创建帧")
                return False
                
        except Exception as e:
            logger.error(f"显示帧时出错: {e}")
            return False

    def packet_receiver_thread(self):
        """用于接收数据包的线程"""
        logger.info("数据包接收线程已启动")
        
        while self.is_running:
            try:
                packet, addr = self.socket.recvfrom(2048) 
                self.process_packet(packet)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"接收数据包时出错: {e}")

    def frame_cleanup_thread(self):
        """用于清理过期帧的线程"""
        logger.info("帧清理线程已启动")
        
        while self.is_running:
            self.cleanup_expired_frames()
            time.sleep(0.5)

    def statistics_thread(self):
        """用于报告统计信息的线程"""
        logger.info("统计信息线程已启动")
        
        while self.is_running:
            time.sleep(5)
            logger.info(f"接收端统计 - Packets: {self.stats['packets_received']}, "
                        f"Frames completed: {self.stats['frames_completed']}, "
                        f"Frames dropped: {self.stats['frames_dropped']}, "
                        f"Out-of-order: {self.stats['packets_out_of_order']}")

    def start_receiving(self):
        """开始视频接收"""
        self.is_running = True
        logger.info("开始接收视频...")
        
        threads = [
            threading.Thread(target=self.packet_receiver_thread, daemon=True),
            threading.Thread(target=self.frame_cleanup_thread, daemon=True),
            threading.Thread(target=self.statistics_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            # 主显示循环
            while self.is_running:
                if self.completed_frames:
                    # 使用锁来安全地从队列中取出元素
                    with self.buffer_lock:
                        if self.completed_frames:
                            frame_info = self.completed_frames.popleft()
                            self.display_frame(frame_info['data'])
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("收到退出信号")
                    break
                
                # 调整休眠时间以匹配期望的帧率，或保持较小的延迟
                time.sleep(1/120) 
                
        except KeyboardInterrupt:
            logger.info("用户中断了接收")
        finally:
            self.stop_receiving()

    def stop_receiving(self):
        """停止视频接收和清理"""
        if self.is_running:
            self.is_running = False
            cv2.destroyAllWindows()
            self.socket.close()
            
            logger.info("视频接收已停止")
            logger.info(f"最终统计 - Packets received: {self.stats['packets_received']}, "
                        f"Frames completed: {self.stats['frames_completed']}, "
                        f"Frames displayed: {self.stats['frames_displayed']}, "
                        f"Frames dropped: {self.stats['frames_dropped']}")

# ***** 修改 main 函数以传递视频尺寸 *****
def main():
    """主函数，运行视频接收器"""
    try:
        # 在这里配置你的视频的实际尺寸
        # 发送端必须发送完全匹配这个尺寸和通道数的原始数据
        receiver = VideoReceiver(
            host='127.0.0.1',
            port=12346,
            frame_timeout=1,
            max_frame_buffer=10,
            video_width=640,       # <--- 在此设置宽度
            video_height=480,      # <--- 在此设置高度
            video_channels=3       # <--- 在此设置通道数 (BGR=3, 灰度=1)
        )
        receiver.start_receiving()
    except Exception as e:
        logger.error(f"main 函数出错: {e}")

if __name__ == "__main__":
    main()