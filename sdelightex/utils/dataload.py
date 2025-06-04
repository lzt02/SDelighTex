import numpy as np
import struct

def load_colmap_data(cameras_path, images_path):
    """加载 COLMAP 相机和图像数据"""
    print("Loading COLMAP camera parameters...")
    if cameras_path.endswith(".txt"):
        camera_params = read_cameras_txt(cameras_path)
    elif cameras_path.endswith(".bin"):
        camera_params = read_cameras_bin(cameras_path)
    else:
        raise ValueError("Unsupported file format: must be .txt or .bin")
    
    print("Loading COLMAP camera poses...")
    if images_path.endswith(".txt"):
        cameras = read_images_text(images_path)
    elif images_path.endswith(".bin"):
        cameras = read_images_bin(images_path)
    else:
        raise ValueError("Unsupported file format: must be .txt or .bin")
    
    print("COLMAP data loaded successfully.")
    return camera_params, cameras
    
def read_cameras_bin(path):
    """读取 COLMAP cameras.bin 文件"""
    camera_models = {
        0: 'SIMPLE_PINHOLE',
        1: 'PINHOLE',
        2: 'SIMPLE_RADIAL',
        3: 'RADIAL',
        4: 'OPENCV',
        5: 'FULL_OPENCV',
    }

    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            model_name = camera_models[model_id]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = {
                0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12
            }[model_id]
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            cameras[camera_id] = {
                'model': model_name,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_images_bin(path):
    """解析 COLMAP 的 images.bin 文件"""
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        images = []

        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            
            # 读取图像文件名
            name = b""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name += char
            name = name.decode("utf-8")

            # 读取点数（但是我们不需要）
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # 跳过 points2D 部分

            images.append({
                'image_id': image_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': camera_id,
                'name': name
            })
        return images

def read_cameras_txt(path):
    """读取 COLMAP cameras.txt 文件"""
    camera_models = {
        0: 'SIMPLE_PINHOLE',
        1: 'PINHOLE',
        2: 'SIMPLE_RADIAL',
        3: 'RADIAL',
        4: 'OPENCV',
        5: 'FULL_OPENCV',
    }

    cameras = {}
    with open(path, "r") as f:
        lines = f.readlines()
        cameras = {}
        for line in lines:
            if line.startswith("#"):
                continue
            data = line.split()
            camera_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = [float(x) for x in data[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_images_text(path):
    """读取 COLMAP images.txt 文件"""
    images = []
    with open(path, "r") as f:
        lines = f.readlines()[4:]
        images = []
        for i in range(0, len(lines), 2):
            if lines[i].startswith("#"):
                continue
            parts = lines[i].strip().split(" ", 9)
            image = {
                'image_id': int(parts[0]),
                'qw': float(parts[1]),
                'qx': float(parts[2]),
                'qy': float(parts[3]),
                'qz': float(parts[4]),
                'tx': float(parts[5]),
                'ty': float(parts[6]),
                'tz': float(parts[7]),
                'camera_id': int(parts[8]),
                'name': parts[9]
            }
            images.append(image)
    return images
    