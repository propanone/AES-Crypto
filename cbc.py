from Crypto.Cipher import AES
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Display image
def display_image(image_path):
    img = Image.open(image_path)
    img.show()

# Visualize R, G, B channels separately with improved plots
def display_channels(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    r_channel, g_channel, b_channel = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cmap_list = ['Reds', 'Greens', 'Blues']
    titles = ["Red Channel", "Green Channel", "Blue Channel"]
    channels = [r_channel, g_channel, b_channel]
    
    for ax, channel, cmap, title in zip(axs, channels, cmap_list, titles):
        img_plot = ax.imshow(channel, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        plt.colorbar(img_plot, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    plt.show()

# File handling
def Open_File(filename):
    with open(filename, "rb") as f:
        return f.read()

def Save_File(filename, block):
    with open(filename, "wb") as f:
        f.write(block)

def Get_Padding(block):
    return (len(block) % 16) * -1

def Encrypt(cipher, read_filename, save_filename):
    block = Open_File(read_filename)
    pad = Get_Padding(block)
    block_trimmed = block[64:pad]
    ciphertext = cipher.encrypt(block_trimmed)
    ciphertext = block[0:64] + ciphertext + block[pad:]
    Save_File(save_filename, ciphertext)

def Decrypt(cipher, read_filename, save_filename):
    block = Open_File(read_filename)
    pad = Get_Padding(block)
    block_trimmed = block[64:pad]
    plaintext = cipher.decrypt(block_trimmed)
    plaintext = block[0:64] + plaintext + block[pad:]
    Save_File(save_filename, plaintext)

def Init_Cipher(key, mode, iv):
    return AES.new(key, mode, iv)

# Improved RGB histogram
def plot_rgb_histogram(image_path, title="RGB Histogram"):
    img = Image.open(image_path)
    img_array = np.array(img)
    r, g, b = img_array[:, :, 0].flatten(), img_array[:, :, 1].flatten(), img_array[:, :, 2].flatten()
    
    plt.figure(figsize=(12, 6))
    plt.hist(r, bins=256, color='red', alpha=0.5, label='Red Channel')
    plt.hist(g, bins=256, color='green', alpha=0.5, label='Green Channel')
    plt.hist(b, bins=256, color='blue', alpha=0.5, label='Blue Channel')
    plt.title(title, fontsize=14)
    plt.xlabel("Pixel Intensity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Calculate RGB Correlation
def calculate_rgb_correlation(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    correlations = {}
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    for i, color in enumerate(['R', 'G', 'B']):
        pairs = {direction: [] for direction in directions}
        height, width = img_array.shape[:2]
        for _ in range(2000):
            i_idx = np.random.randint(0, height - 1)
            j_idx = np.random.randint(0, width - 1)
            pairs['Horizontal'].append((img_array[i_idx, j_idx, i], img_array[i_idx, j_idx + 1, i]))
            pairs['Vertical'].append((img_array[i_idx, j_idx, i], img_array[i_idx + 1, j_idx, i]))
            pairs['Diagonal'].append((img_array[i_idx, j_idx, i], img_array[i_idx + 1, j_idx + 1, i]))
        correlations[color] = {dir_: np.corrcoef(*zip(*pairs[dir_]))[0, 1] for dir_ in directions}
    return correlations

# NPCR & UACI
def calculate_npcr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    diff = np.sum(img1 != img2)
    total_pixels = img1.size
    return (diff / total_pixels) * 100

def calculate_uaci(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path), dtype=np.float32)
    img2 = np.array(Image.open(img2_path), dtype=np.float32)
    diff = np.abs(img1 - img2)
    return np.sum(diff) / (img1.size * 255) * 100

# Plot difference histogram
def plot_difference_histogram(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    diff = np.abs(img1 - img2).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(diff, bins=256, color='purple', alpha=0.7, label='Difference')
    plt.title("Difference Histogram", fontsize=14)
    plt.xlabel("Pixel Intensity Difference", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Main execution
key = b"aaaabbbbccccdddd"
iv = b"1111222233334444"
mode = AES.MODE_CBC

cipher_encrypt = Init_Cipher(key, mode, iv)
cipher_decrypt = Init_Cipher(key, mode, iv)

Encrypt(cipher_encrypt, "baboon.bmp", "e_baboon_cbc.bmp")
Decrypt(cipher_decrypt, "e_baboon_cbc.bmp", "d_baboon_cbc.bmp")

# Display results
display_channels("baboon.bmp")
plot_rgb_histogram("baboon.bmp", "Original Image RGB Histogram")
plot_rgb_histogram("e_baboon_cbc.bmp", "Encrypted Image RGB Histogram")

original_corr = calculate_rgb_correlation("baboon.bmp")
encrypted_corr = calculate_rgb_correlation("e_baboon_cbc.bmp")
print("Original Correlations:", original_corr)
print("Encrypted Correlations:", encrypted_corr)
