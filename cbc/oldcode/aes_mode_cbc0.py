from Crypto.Cipher import AES
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def display_image(image_path):
    img = Image.open(image_path)
    img.show()

# Visualize R, G, B channels separately

def display_channels(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    # Extract R, G, B channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]

    # Display each channel
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(r_channel, cmap='Reds')
    plt.title("Red Channel")
    plt.subplot(1, 3, 2)
    plt.imshow(g_channel, cmap='Greens')
    plt.title("Green Channel")
    plt.subplot(1, 3, 3)
    plt.imshow(b_channel, cmap='Blues')
    plt.title("Blue Channel")
    plt.show()


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

def Encrypt(cipher,read_filename, save_filename):
    block = Open_File(read_filename)
    pad = Get_Padding(block)
    block_trimmed = block[64:pad]
    ciphertext = cipher.encrypt(block_trimmed)
    ciphertext = block[0:64] + ciphertext + block[pad:]
    Save_File(save_filename, ciphertext) 

def Decrypt(cipher,read_filename, save_filename):
    block = Open_File(read_filename)
    pad = Get_Padding(block)
    block_trimmed = block[64:pad]
    ciphertext = cipher.decrypt(block_trimmed)
    ciphertext = block[0:64] + ciphertext + block[pad:]
    Save_File(save_filename, ciphertext) 

def Init_Cipher(key, mode, iv):
    cipher = AES.new(key, mode, iv)
    return cipher



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



# Function to create a modified image
def create_modified_image(image_path, output_path):
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.uint8)

    # Modify one pixel (flip the top-left corner)
    img_array[0, 0, 0] = np.uint8(int(img_array[0, 0, 0] + 1) % 256)  # R channel
    img_array[0, 0, 1] = np.uint8(int(img_array[0, 0, 1] + 1) % 256)  # G channel
    img_array[0, 0, 2] = np.uint8(int(img_array[0, 0, 2] + 1) % 256)  # B channel

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

def plot_difference_histogram(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))
    
    # Flatten the images to 1D arrays for histogram comparison
    diff = np.abs(img2 - img1)
    diff_flat = diff.flatten()
    
    # Plot the histogram
    plt.hist(diff_flat, bins=256, range=(0, 255), density=True, color='b', alpha=0.7, label='Difference Histogram')
    plt.title("Difference Histogram between Encrypted and Modified Images")
    plt.xlabel("Pixel Intensity Difference")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_rgb_correlation_scatter(image_path, title_prefix="Original"):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"{title_prefix} - RGB Correlation Scatter Plots", fontsize=16)
    
    for channel_idx, channel in enumerate(channel_names):
        for dir_idx, direction in enumerate(directions):  # Use dir_idx instead of overwriting i
            x = []
            y = []
            height, width = img_array.shape[:2]
            for _ in range(2000):  # Sample 2000 random points
                i = np.random.randint(0, height - 1)
                j = np.random.randint(0, width - 1)
                
                if direction == 'Horizontal' and j < width - 1:
                    x.append(img_array[i, j, channel_idx])
                    y.append(img_array[i, j + 1, channel_idx])
                elif direction == 'Vertical' and i < height - 1:
                    x.append(img_array[i, j, channel_idx])
                    y.append(img_array[i + 1, j, channel_idx])
                elif direction == 'Diagonal' and i < height - 1 and j < width - 1:
                    x.append(img_array[i, j, channel_idx])
                    y.append(img_array[i + 1, j + 1, channel_idx])
            
            ax = axs[channel_idx, dir_idx]  # Access correct subplot using dir_idx
            ax.scatter(x, y, s=1, color=colors[channel_idx], alpha=0.5)
            ax.set_title(f"{channel} - {direction}")
            ax.set_xlabel("Pixel Intensity (X)")
            ax.set_ylabel("Pixel Intensity (Y)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle spacing
    plt.show()


png_image = Image.open("baboon.png")
png_image.save("baboon.bmp", format="BMP")


 # set the key and iv values
key = b"aaaabbbbccccdddd"
iv = b"1111222233334444"

 # Available AES Block Modes
 # AES.MODE_ECB = 1
 # AES.MODE_CBC = 2
 # AES.MODE_CFB = 3
 # AES.MODE_OFB = 5
 # AES.MODE_CTR = 6
 # AES.MODE_OPENPGP = 7

#mode = AES.MODE_OFB
#mode = AES.MODE_CFB
mode = AES.MODE_CBC
c = Init_Cipher(key,mode, iv)


Encrypt(c, "baboon.bmp", "e_baboon_cbc.bmp")
d = Init_Cipher(key, mode, iv)
Decrypt(d, "e_baboon_cbc.bmp", "d_baboon_cbc.bmp")

print("done")


display_channels("baboon.bmp")

#Plot RGB histograms for the original and encrypted images
#plot_rgb_histogram("baboon.bmp", "Original Image RGB Histogram")
#plot_rgb_histogram("e_baboon_cbc.bmp", "Encrypted Image RGB Histogram")

## Compute correlations for RGB
original_rgb_corr = calculate_rgb_correlation("baboon.bmp")
encrypted_rgb_corr = calculate_rgb_correlation("e_baboon_cbc.bmp")

print("Original RGB Correlations:", original_rgb_corr)
print("Encrypted RGB Correlations:", encrypted_rgb_corr)

# Plot correlation scatter plots for Original and Encrypted images
plot_rgb_correlation_scatter("baboon.bmp", "Original Image")
plot_rgb_correlation_scatter("e_baboon_cbc.bmp", "Encrypted Image")



# Compute entropy for original and encrypted images
original_entropy = calculate_rgb_entropy("baboon.bmp")
encrypted_entropy = calculate_rgb_entropy("e_baboon_cbc.bmp")

print("Original Image Entropy (R, G, B):", original_entropy)
print("Encrypted Image Entropy (R, G, B):", encrypted_entropy)


# differntial attack NPCR and UACI values 

# Step 1: Create a modified plaintext image
create_modified_image("baboon.bmp", "baboon_modified.bmp")

# Step 2: Encrypt  and modified images
Encrypt(c, "baboon_modified.bmp", "e_baboon_modified_cbc.bmp")

# Step 3: Compute NPCR and UACI
npcr_value = calculate_npcr("e_baboon_cbc.bmp", "e_baboon_modified_cbc.bmp")
uaci_value = calculate_uaci("e_baboon_cbc.bmp", "e_baboon_modified_cbc.bmp")

print(f"NPCR: {npcr_value:.2f}%")
print(f"UACI: {uaci_value:.2f}%")

plot_rgb_histogram("e_baboon_cbc.bmp", "Encrypted Image RGB Histogram")
plot_rgb_histogram("e_baboon_modified_cbc.bmp", "Modified-Encrypted Image RGB Histogram")
plot_difference_histogram("e_baboon_cbc.bmp" ,"e_baboon_modified_cbc.bmp" )