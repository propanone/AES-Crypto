from Crypto.Cipher import AES
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from Crypto.Cipher import AES

#  encryption function
def ecb_encrypt(input_file, output_file, key):
    cipher = AES.new(key, AES.MODE_ECB)

    with open(input_file, "rb") as f:
        byteblock = f.read()

    # Calculate padding and trim
    pad = len(byteblock) % 16 * -1
    byteblock_trimmed = byteblock[64:pad]

    # Encrypt trimmed block
    ciphertext = cipher.encrypt(byteblock_trimmed)

    # Combine header and footer with ciphertext
    ciphertext = byteblock[0:64] + ciphertext + byteblock[pad:]

    with open(output_file, "wb") as f:
        f.write(ciphertext)

    print(f"File encrypted and saved to {output_file}")



# decryption function
def ecb_decrypt(input_file, output_file, key):
    cipher = AES.new(key, AES.MODE_ECB)

    with open(input_file, "rb") as f:
        byteblock = f.read()

    # Calculate padding and trim
    pad = len(byteblock) % 16 * -1
    byteblock_trimmed = byteblock[64:pad]

    # Decrypt trimmed block
    plaintext = cipher.decrypt(byteblock_trimmed)

    # Combine header and footer with plaintext
    plaintext = byteblock[0:64] + plaintext + byteblock[pad:]

    with open(output_file, "wb") as f:
        f.write(plaintext)

    print(f"File decrypted and saved to {output_file}")


def calculate_correlation(image_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    height, width = img_array.shape
    pairs = []

    # Extract adjacent pixel pairs (horizontal, vertical, diagonal)
    for _ in range(2000):
        i = np.random.randint(0, height - 1)
        j = np.random.randint(0, width - 1)
        pairs.append((
            (img_array[i, j], img_array[i, j + 1]),  # Horizontal
            (img_array[i, j], img_array[i + 1, j]),  # Vertical
            (img_array[i, j], img_array[i + 1, j + 1])  # Diagonal
        ))

    correlations = []
    for pair_type in zip(*pairs):
        x, y = zip(*pair_type)
        correlation = np.corrcoef(x, y)[0, 1]
        correlations.append(correlation)
    return correlations

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



def plot_rgb_histogram(image_path, title="RGB Histogram"):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Separate the RGB channels
    r_channel = img_array[:, :, 0].flatten()  # Red channel
    g_channel = img_array[:, :, 1].flatten()  # Green channel
    b_channel = img_array[:, :, 2].flatten()  # Blue channel

    # Plot histograms for each channel
    plt.figure(figsize=(8, 6))
    plt.hist(r_channel, bins=256, range=(0, 256), color='red', alpha=0.6, label='Red Channel')
    plt.hist(g_channel, bins=256, range=(0, 256), color='green', alpha=0.6, label='Green Channel')
    plt.hist(b_channel, bins=256, range=(0, 256), color='blue', alpha=0.6, label='Blue Channel')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Function to create a modified image
def create_modified_image(image_path, output_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    # Modify one pixel (flip the top-left corner)
    img_array[0, 0, 0] = (img_array[0, 0, 0] + 1) % 256  # R channel
    img_array[0, 0, 1] = (img_array[0, 0, 1] + 1) % 256  # G channel
    img_array[0, 0, 2] = (img_array[0, 0, 2] + 1) % 256  # B channel

    # Save the modified image
    modified_img = Image.fromarray(img_array)
    modified_img.save(output_path)


# Function to calculate NPCR
def calculate_npcr(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))

    # Calculate D(i, j)
    diff = np.sum(img1 != img2)
    total_pixels = img1.shape[0] * img1.shape[1] * img1.shape[2]  # Total pixels (RGB)

    npcr = (diff / total_pixels) * 100
    return npcr

# Function to calculate UACI
def calculate_uaci(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path), dtype=np.float32)
    img2 = np.array(Image.open(image2_path), dtype=np.float32)

    # Calculate the absolute difference
    diff = np.abs(img1 - img2)
    uaci = np.sum(diff) / (img1.shape[0] * img1.shape[1] * img1.shape[2] * 255) * 100
    return uaci


png_image = Image.open("baboon.png")
png_image.save("baboon.bmp", format="BMP")

key = b"aaaabbbbccccdddd"
cipher = AES.new(key, AES.MODE_ECB)


print ("done")

# Encrypt the image
ecb_encrypt("baboon.bmp", "e_baboon_ecb.bmp", key)

# Decrypt the image
ecb_decrypt("e_baboon_ecb.bmp", "d_baboon_ecb.bmp", key)

# Plot RGB histograms for the original and encrypted images
plot_rgb_histogram("baboon.bmp", "Original Image RGB Histogram")
plot_rgb_histogram("e_baboon_ecb.bmp", "Encrypted Image RGB Histogram")

# Compute correlations
original_corr = calculate_correlation("baboon.bmp")
encrypted_corr = calculate_correlation("e_baboon_ecb.bmp")

print(f"Original Image Correlations (Horizontal, Vertical, Diagonal): {original_corr}")
print(f"Encrypted Image Correlations (Horizontal, Vertical, Diagonal): {encrypted_corr}")

# Compute correlations for RGB
original_rgb_corr = calculate_rgb_correlation("baboon.bmp")
encrypted_rgb_corr = calculate_rgb_correlation("e_baboon_ecb.bmp")

print("Original RGB Correlations:", original_rgb_corr)
print("Encrypted RGB Correlations:", encrypted_rgb_corr)

# Compute entropy for original and encrypted images
original_entropy = calculate_rgb_entropy("baboon.bmp")
encrypted_entropy = calculate_rgb_entropy("e_baboon_ecb.bmp")

print("Original Image Entropy (R, G, B):", original_entropy)
print("Encrypted Image Entropy (R, G, B):", encrypted_entropy)


# NPCR and UACI values
# Step 1: Create a modified plaintext image
create_modified_image("baboon.bmp", "baboon_modified.bmp")

# Step 2: Encrypt  and modified images
ecb_encrypt("baboon_modified.bmp", "e_baboon_modified_ecb.bmp",key)

# Step 3: Compute NPCR and UACI
npcr_value = calculate_npcr("e_baboon_ecb.bmp", "e_baboon_modified_ecb.bmp")
uaci_value = calculate_uaci("e_baboon_ecb.bmp", "e_baboon_modified_ecb.bmp")

print(f"NPCR: {npcr_value:.2f}%")
print(f"UACI: {uaci_value:.2f}%")