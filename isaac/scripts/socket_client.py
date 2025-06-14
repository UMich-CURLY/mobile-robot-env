import socket
import pickle
import jsonpickle
import json
import time
import numpy as np
import cv2
import struct

# --- Configuration ---
# SERVER_HOST = '35.3.201.75' # IP address of the sensor machine (use 'localhost' if running on the same machine)'localhost'#
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345
REQUEST_MESSAGE = b"GET_SENSOR_DATA"
REQUEST_INTERVAL_SEC = 1.0 # 1 Hz



def recv_all(sock, n):
    """Helper function to receive n bytes from a socket."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None # Connection closed
        data.extend(packet)
    return data

def decompress_payload(compressed_payload_dict):
    """
    Decompresses 'rgb_image' and 'depth_image' in the payload dictionary
    if they were PNG-compressed.

    Args:
        compressed_payload_dict (dict): The dictionary received from the server,
                                        potentially with compressed image data as bytes.

    Returns:
        dict: A new dictionary with image bytes replaced by NumPy arrays.
    """
    decompressed_dict = compressed_payload_dict.copy()

    # Decompress RGB Image
    if decompressed_dict.get('rgb_image_compressed_format') == 'png' and \
       isinstance(decompressed_dict.get('rgb_image'), bytes):
        encoded_bytes = decompressed_dict['rgb_image']
        # Convert bytes back to NumPy array for imdecode
        np_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
        rgb_image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if rgb_image_np is not None:
            decompressed_dict['rgb_image'] = rgb_image_np
            # Optionally, verify against stored shape/dtype if they were sent
            # stored_shape = decompressed_dict.get('rgb_image_shape')
            # stored_dtype = decompressed_dict.get('rgb_image_dtype')
            # if stored_shape and rgb_image_np.shape != stored_shape:
            #     print(f"Warning: Decompressed RGB shape {rgb_image_np.shape} differs from original {stored_shape}")
            # if stored_dtype and str(rgb_image_np.dtype) != stored_dtype:
            #     print(f"Warning: Decompressed RGB dtype {rgb_image_np.dtype} differs from original {stored_dtype}")
        else:
            print("Warning: RGB image PNG decoding failed.")
            decompressed_dict['rgb_image'] = None # Or handle error
        # Clean up compression-specific keys
        decompressed_dict.pop('rgb_image_compressed_format', None)
        decompressed_dict.pop('rgb_image_shape', None)
        decompressed_dict.pop('rgb_image_dtype', None)


    # Decompress Depth Image
    if decompressed_dict.get('depth_image_compressed_format') == 'png' and \
       isinstance(decompressed_dict.get('depth_image'), bytes):
        encoded_bytes = decompressed_dict['depth_image']
        np_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
        # cv2.IMREAD_UNCHANGED is crucial for 16-bit depth images
        depth_image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_image_np is not None:
            decompressed_dict['depth_image'] = depth_image_np
        else:
            print("Warning: Depth image PNG decoding failed.")
            decompressed_dict['depth_image'] = None
        # Clean up compression-specific keys
        decompressed_dict.pop('depth_image_compressed_format', None)
        decompressed_dict.pop('depth_image_shape', None)
        decompressed_dict.pop('depth_image_dtype', None)

    return decompressed_dict
def send_action_message(msg,host = SERVER_HOST):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0) # Timeout for connection and operations
    client_socket.connect((host, SERVER_PORT))
    # print("Connected to server.")
    message = msg.type+" "+jsonpickle.encode(msg)
    client_socket.sendall(message.encode())

def request_sensor_data(host = SERVER_HOST):
    # print(f"\n[{time.strftime('%H:%M:%S')}] Attempting to connect to {SERVER_HOST}:{SERVER_PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0) # Timeout for connection and operations
    client_socket.connect((host, SERVER_PORT))
    # print("Connected to server.")

    # 1. Send request
    client_socket.sendall(REQUEST_MESSAGE)
    # print(f"Sent request: {REQUEST_MESSAGE.decode()}")

    # 2. Receive the length of the pickled data (8 bytes, unsigned long long)
    raw_msglen = recv_all(client_socket, 8)
    if not raw_msglen:
        print("Connection closed by server before sending data length.")
        return None 
    
    msglen = struct.unpack('>Q', raw_msglen)[0]
    # print(f"Expecting pickled data of length: {msglen} bytes")

    # 3. Receive the pickled data
    pickled_payload = recv_all(client_socket, msglen)
    if not pickled_payload:
        print("Connection closed by server before sending full payload.")
        return None

    # 4. Deserialize
    payload = pickle.loads(pickled_payload)
    payload = decompress_payload(payload)

    return payload

def request_planner_state(host = SERVER_HOST):
    # print(f"\n[{time.strftime('%H:%M:%S')}] Attempting to connect to {SERVER_HOST}:{SERVER_PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0) # Timeout for connection and operations
    client_socket.connect((host, SERVER_PORT))
    # print("Connected to server.")

    # 1. Send request
    client_socket.sendall(b"GET_PLANNER_STATE")
    # print(f"Sent request: {REQUEST_MESSAGE.decode()}")

    # 2. Receive the length of the pickled data (8 bytes, unsigned long long)
    raw_msglen = recv_all(client_socket, 8)
    if not raw_msglen:
        print("Connection closed by server before sending data length.")
        return None 
    
    msglen = struct.unpack('>Q', raw_msglen)[0]
    # print(f"Expecting pickled data of length: {msglen} bytes")

    # 3. Receive the pickled data
    json_payload = recv_all(client_socket, msglen).decode()
    if not json_payload:
        print("Connection closed by server before sending full payload.")
        return None

    # 4. Deserialize
    payload = json.loads(json_payload)
    return payload
def main():
    print("Agent Socket Client Started")
    while True:
        start_cycle_time = time.time()
        payload = None # Initialize payload to None for this cycle

        try:
            payload = request_sensor_data()

            # print("Successfully received and unpickled data.")
            
            # --- Process Data ---
            if payload and payload.get("success", False):
                rgb_image = payload.get("rgb_image")
                depth_image = payload.get("depth_image")
                pose = payload.get("pose")
                server_timestamp_ns = payload.get("timestamp_server_ns")
                
                latency_ms = (time.time_ns() - server_timestamp_ns) / 1_000_000 if server_timestamp_ns else -1
                if(latency_ms>500):
                    print(f"  Server timestamp (ns): {server_timestamp_ns}, Approx E2E Latency: {latency_ms:.2f} ms")

                if rgb_image is not None:
                    print(f"  RGB Image: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                else:
                    print("  RGB Image: None")

                if depth_image is not None:
                    print(f"  Depth Image: {depth_image.shape}, dtype: {depth_image.dtype}")
                    # Normalize depth for display (example: if 16UC1 in mm, display 0-5m)
                    if depth_image.dtype == np.uint16:
                        depth_display = np.clip(depth_image, 0, 5000) # 0 to 5 meters
                        depth_display = (depth_display / 5000.0 * 255).astype(np.uint8)
                    elif depth_image.dtype == np.float32: # Assuming meters
                        depth_display = np.clip(depth_image, 0, 5.0)
                        depth_display = (depth_display / 5.0 * 255).astype(np.uint8)
                    else:
                        print(f"  Cannot display depth image with dtype {depth_image.dtype} automatically.")
                else:
                    print("  Depth Image: None")

                if pose:
                    p = pose['pose']['position']
                    o = pose['pose']['orientation']
                    print(f"  Pose ({pose['header']['frame_id']} @ {pose['header']['stamp_sec']}.{pose['header']['stamp_nanosec']}):")
                    print(f"    Pos: x={p['x']:.2f}, y={p['y']:.2f}, z={p['z']:.2f}")
                    print(f"    Ori: x={o['x']:.2f}, y={o['y']:.2f}, z={o['z']:.2f}, w={o['w']:.2f}")
                else:
                    print("  Pose: None")
                
                if cv2.waitKey(1) & 0xFF == ord('q'): # Allow quitting via OpenCV window
                    break
            elif payload: # success == False or no success key
                print(f"  Server responded with an issue: {payload.get('message', 'Unknown error')}")
            else: # Should not happen if recv_all worked
                print("  Received empty or invalid payload from server.")


        except socket.timeout:
            print(f"Socket timeout during operation with {SERVER_HOST}:{SERVER_PORT}")
        except socket.error as e:
            print(f"Socket error: {e}")
        except pickle.UnpicklingError as e:
            print(f"Pickle error: {e}. Received data might be corrupt or not a pickle.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            if 'client_socket' in locals():
                client_socket.close()
            
            # Maintain 1Hz cycle
            elapsed_cycle_time = time.time() - start_cycle_time
            sleep_time = REQUEST_INTERVAL_SEC - elapsed_cycle_time
            if sleep_time > 0:
                # print(f"Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

    cv2.destroyAllWindows()
    print("Agent Socket Client Shutting Down.")

if __name__ == '__main__':
    main()