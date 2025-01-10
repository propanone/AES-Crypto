from PIL import Image
from Crypto.Cipher import AES
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Create an all-white image
white_image = Image.new("RGB", (256, 256), (255, 255, 255))  # 256x256 white image
white_image.save("all_white.bmp")

# Create an all-black image
black_image = Image.new("RGB", (256, 256), (0, 0, 0))  # 256x256 black image
black_image.save("all_black.bmp")


def encrypt_ecb(input_path, output_path, key):
    # Load the image
    with open(input_path, "rb") as f:
        byteblock = f.read()

    # Trim padding to make it divisible by 16
    pad = len(byteblock) % 16 * -1
    byteblock_trimmed = byteblock[64:pad]

    # Encrypt
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(byteblock_trimmed)
    ciphertext = byteblock[0:64] + ciphertext + byteblock[pad:]

    # Save encrypted image
    with open(output_path, "wb") as f:
        f.write(ciphertext)

def Open_File(filename):
    with open(filename, "rb") as f:
        byteblock = f.read()
    return byteblock
def Save_File(filename, block):
    with open(filename,"wb") as f:
        f.write(block)
def Get_Padding(block):
    l = len(block) %16
    return (l * -1)

def Encrypt_cbc(cipher,read_filename, save_filename):
    block = Open_File(read_filename)
    pad = Get_Padding(block)
    block_trimmed = block[64:pad]
    ciphertext = cipher.encrypt(block_trimmed)
    ciphertext = block[0:64] + ciphertext + block[pad:]
    Save_File(save_filename, ciphertext) 

def Decrypt_cbc(cipher,read_filename, save_filename):
    block = Open_File(read_filename)
    pad = Get_Padding(block)
    block_trimmed = block[64:pad]
    ciphertext = cipher.decrypt(block_trimmed)
    ciphertext = block[0:64] + ciphertext + block[pad:]
    Save_File(save_filename, ciphertext) 

def Init_Cipher(key, mode, iv):
    cipher = AES.new(key, mode, iv)
    return cipher

def plot_histogram(image_path, title):
    img = np.array(Image.open(image_path))
    r, g, b = img[..., 0], img[..., 1], img[..., 2]  # Split RGB channels

    plt.figure(figsize=(10, 4))
    plt.hist(r.ravel(), bins=256, color="red", alpha=0.6, label="Red Channel")
    plt.hist(g.ravel(), bins=256, color="green", alpha=0.6, label="Green Channel")
    plt.hist(b.ravel(), bins=256, color="blue", alpha=0.6, label="Blue Channel")
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def calculate_rgb_correlation(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)  # RGB image
    correlations = {"R": [], "G": [], "B": []}

    for channel, color in enumerate(["R", "G", "B"]):
        channel_data = img_array[:, :, channel]  # Extract specific channel
        height, width = channel_data.shape
        pairs = []

        # Extract adjacent pixel pairs (Horizontal, Vertical, Diagonal)
        for _ in range(2000):
            i = np.random.randint(0, height - 1)
            j = np.random.randint(0, width - 1)
            pairs.append((
                (channel_data[i, j], channel_data[i, j + 1]),  # Horizontal
                (channel_data[i, j], channel_data[i + 1, j]),  # Vertical
                (channel_data[i, j], channel_data[i + 1, j + 1])  # Diagonal
            ))

        # Compute correlations for Horizontal, Vertical, Diagonal
        for pair_type in zip(*pairs):
            x, y = zip(*pair_type)
            correlation = np.corrcoef(x, y)[0, 1]
            correlations[color].append(correlation)

    return correlations



def calculate_rgb_entropy(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)  # Convert image to array
    entropy = {"R": 0, "G": 0, "B": 0}

    for channel, color in enumerate(["R", "G", "B"]):
        channel_data = img_array[:, :, channel].flatten()  # Flatten channel
        pixel_counts = Counter(channel_data)
        total_pixels = len(channel_data)
        probabilities = [count / total_pixels for count in pixel_counts.values()]
        entropy[color] = -sum(p * np.log2(p) for p in probabilities if p > 0)

    return entropy

key = b"aaaabbbbccccdddd"
iv = b"1111222233334444"

mode = AES.MODE_CBC
c = Init_Cipher(key,mode, iv)


Encrypt_cbc(c, "all_white.bmp", "encrypted_white_cbc.bmp")
c = Init_Cipher(key, mode, iv)
Encrypt_cbc(c, "all_black.bmp", "encrypted_black_cbc.bmp")



encrypt_ecb("all_white.bmp", "encrypted_white_ecb.bmp", key)
encrypt_ecb("all_black.bmp", "encrypted_black_ecb.bmp", key)

# Plot histograms for encrypted white and black images cbc mode 
plot_histogram("encrypted_white_cbc.bmp", "Histogram of Encrypted All-White Image cbc")
plot_histogram("encrypted_black_cbc.bmp", "Histogram of Encrypted All-Black Image cbc")

# Plot histograms for encrypted white and black images ebc mode 
plot_histogram("encrypted_white_ecb.bmp", "Histogram of Encrypted All-White Image ecb")
plot_histogram("encrypted_black_ecb.bmp", "Histogram of Encrypted All-Black Image ecb")


## Compute correlations for RGB
original_rgb_corr = calculate_rgb_correlation("all_white.bmp")
encrypted_rgb_corr = calculate_rgb_correlation("encrypted_white_cbc.bmp")

print("Original RGB Correlations:", original_rgb_corr)
print("Encrypted RGB Correlations:", encrypted_rgb_corr)


# Compute entropy for original and encrypted images
original_entropy = calculate_rgb_entropy("all_white.bmp")
encrypted_entropy = calculate_rgb_entropy("encrypted_white_cbc.bmp")

print("Original Image Entropy (R, G, B):", original_entropy)
print("Encrypted Image Entropy (R, G, B):", encrypted_entropy)