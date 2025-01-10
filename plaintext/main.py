import os
from PIL import Image
from Crypto.Cipher import AES
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Create output directories
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/text", exist_ok=True)
os.makedirs("results/images", exist_ok=True)
os.makedirs("results/encrypted", exist_ok=True)


# Create an all-white image
white_image = Image.new("RGB", (256, 256), (255, 255, 255))
white_image.save("results/images/all_white.bmp")

# Create an all-black image
black_image = Image.new("RGB", (256, 256), (0, 0, 0))
black_image.save("results/images/all_black.bmp")

input_white = "results/images/all_white.bmp"
input_black = "results/images/all_black.bmp"

# Encryption and decryption functions
def encrypt_ecb(input_path, output_path, key):
    with open(input_path, "rb") as f:
        byteblock = f.read()

    pad = len(byteblock) % 16 * -1
    byteblock_trimmed = byteblock[64:pad]

    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(byteblock_trimmed)
    ciphertext = byteblock[0:64] + ciphertext + byteblock[pad:]

    with open(output_path, "wb") as f:
        f.write(ciphertext)

def open_file(filename):
    with open(filename, "rb") as f:
        return f.read()

def save_file(filename, block):
    with open(filename, "wb") as f:
        f.write(block)

def get_padding(block):
    return (len(block) % 16) * -1

def encrypt_cbc(cipher, input_path, output_path):
    block = open_file(input_path)
    pad = get_padding(block)
    block_trimmed = block[64:pad]
    ciphertext = cipher.encrypt(block_trimmed)
    ciphertext = block[0:64] + ciphertext + block[pad:]
    save_file(output_path, ciphertext)

def decrypt_cbc(cipher, input_path, output_path):
    block = open_file(input_path)
    pad = get_padding(block)
    block_trimmed = block[64:pad]
    plaintext = cipher.decrypt(block_trimmed)
    plaintext = block[0:64] + plaintext + block[pad:]
    save_file(output_path, plaintext)

def init_cipher(key, mode, iv=None):
    return AES.new(key, mode, iv) if iv else AES.new(key, mode)

# Visualization functions
def plot_histogram(image_path, title, output_path):
    img = np.array(Image.open(image_path))
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    plt.figure(figsize=(10, 4))
    plt.hist(r.ravel(), bins=256, color="red", alpha=0.6, label="Red Channel")
    plt.hist(g.ravel(), bins=256, color="green", alpha=0.6, label="Green Channel")
    plt.hist(b.ravel(), bins=256, color="blue", alpha=0.6, label="Blue Channel")
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def calculate_rgb_correlation(image_path):
    img_array = np.array(Image.open(image_path))
    correlations = {"R": [], "G": [], "B": []}

    for channel, color in enumerate(["R", "G", "B"]):
        channel_data = img_array[:, :, channel]
        height, width = channel_data.shape
        pairs = []

        for _ in range(2000):
            i = np.random.randint(0, height - 1)
            j = np.random.randint(0, width - 1)
            pairs.append(((channel_data[i, j], channel_data[i, j + 1]), (channel_data[i, j], channel_data[i + 1, j]), (channel_data[i, j], channel_data[i + 1, j + 1])))

        for pair_type in zip(*pairs):
            x, y = zip(*pair_type)
            if np.std(x) > 0 and np.std(y) > 0:  # Check to avoid division by zero
                correlation = np.corrcoef(x, y)[0, 1]
                if not np.isnan(correlation):
                    correlations[color].append(correlation)

    return correlations

def calculate_rgb_entropy(image_path):
    img_array = np.array(Image.open(image_path))
    entropy = {"R": 0, "G": 0, "B": 0}

    for channel, color in enumerate(["R", "G", "B"]):
        channel_data = img_array[:, :, channel].flatten()
        pixel_counts = Counter(channel_data)
        total_pixels = len(channel_data)
        probabilities = [count / total_pixels for count in pixel_counts.values()]
        entropy[color] = -sum(p * np.log2(p) for p in probabilities if p > 0)

    return entropy

# Encryption keys
key = b"aaaabbbbccccdddd"
iv = b"1111222233334444"

# Encrypt images using CBC mode
cipher_cbc = init_cipher(key, AES.MODE_CBC, iv)
encrypt_cbc(cipher_cbc, input_white, "results/encrypted/encrypted_white_cbc.bmp")

cipher_cbc = init_cipher(key, AES.MODE_CBC, iv)
encrypt_cbc(cipher_cbc, input_black, "results/encrypted/encrypted_black_cbc.bmp")

# Encrypt images using ECB mode
encrypt_ecb(input_white, "results/encrypted/encrypted_white_ecb.bmp", key)
encrypt_ecb(input_black, "results/encrypted/encrypted_black_ecb.bmp", key)

# Plot histograms for encrypted images (CBC mode)
plot_histogram("results/encrypted/encrypted_white_cbc.bmp", "Histogram of Encrypted All-White Image (CBC)", "results/plots/encrypted_white_cbc.png")
plot_histogram("results/encrypted/encrypted_black_cbc.bmp", "Histogram of Encrypted All-Black Image (CBC)", "results/plots/encrypted_black_cbc.png")

# Plot histograms for encrypted images (ECB mode)
plot_histogram("results/encrypted/encrypted_white_ecb.bmp", "Histogram of Encrypted All-White Image (ECB)", "results/plots/encrypted_white_ecb.png")
plot_histogram("results/encrypted/encrypted_black_ecb.bmp", "Histogram of Encrypted All-Black Image (ECB)", "results/plots/encrypted_black_ecb.png")

# Calculate and write correlations to a text file
original_correlation = calculate_rgb_correlation(input_white)
encrypted_correlation_cbc = calculate_rgb_correlation("results/encrypted/encrypted_white_cbc.bmp")

with open("results/text/correlations.txt", "w") as f:
    f.write("Original RGB Correlations:\n")
    f.write(str(original_correlation) + "\n\n")
    f.write("Encrypted RGB Correlations (CBC):\n")
    f.write(str(encrypted_correlation_cbc) + "\n")

# Calculate and write entropy to a text file
original_entropy = calculate_rgb_entropy(input_white)
encrypted_entropy_cbc = calculate_rgb_entropy("results/encrypted/encrypted_white_cbc.bmp")

with open("results/text/entropy.txt", "w") as f:
    f.write("Original Image Entropy (R, G, B):\n")
    f.write(str(original_entropy) + "\n\n")
    f.write("Encrypted Image Entropy (CBC, R, G, B):\n")
    f.write(str(encrypted_entropy_cbc) + "\n")

print("All tasks completed successfully. Results saved in the 'results' directory.")