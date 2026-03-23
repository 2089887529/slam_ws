import socket
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 7788))
sock.settimeout(5)

print("等待 DIFOP 包...")
data, addr = sock.recvfrom(1248)

imu_data = data[1092:1120]
qx, qy, qz, qw = struct.unpack('>ffff', imu_data[0:16])
x,  y,  z      = struct.unpack('>fff',  imu_data[16:28])

print(f"四元数: qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}, qw={qw:.6f}")
print(f"位移:   x={x:.6f},  y={y:.6f},  z={z:.6f}")