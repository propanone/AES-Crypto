from Crypto.Cipher import AES
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime
import glob
import math


class ImageEncryptionAnalyzer:
    def __init__(self, key: bytes = b"aaaabbbbccccdddd", iv: bytes = b"1111222233334444"):
        self.key = key
        self.iv = iv
        self.mode = AES.MODE_CBC
        self.results_dir = self._create_results_directory()

    def _create_results_directory(self) -> str:
        """Create a timestamped directory for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        return results_dir

    def save_plot(self, filename: str):
        """Save plot to results directory."""
        plt.savefig(os.path.join(self.results_dir, "plots", filename), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def init_cipher(self) -> AES:
        """Initialize AES cipher."""
        cipher = AES.new(self.key, self.mode, self.iv)
        return cipher

    def process_file(self, filename: str) -> bytes:
        """Read file and return bytes."""
        with open(filename, "rb") as f:
            return f.read()

    def save_file(self, filename: str, data: bytes) -> None:
        """Save bytes to file."""
        filepath = os.path.join(self.results_dir, "images", os.path.basename(filename))
        with open(filepath, "wb") as f:
            f.write(data)

    def get_padding(self, block: bytes) -> int:
        """Calculate padding for AES block size."""
        return (len(block) % 16) * -1

    def encrypt_image(self, cipher: AES, input_path: str, output_path: str) -> None:
        """Encrypt image using AES-CBC."""
        block = self.process_file(input_path)
        pad = self.get_padding(block)
        block_trimmed = block[64:pad]
        ciphertext = cipher.encrypt(block_trimmed)
        ciphertext = block[0:64] + ciphertext + block[pad:]
        self.save_file(output_path, ciphertext)

    def decrypt_image(self, cipher: AES, input_path: str, output_path: str) -> None:
        """Decrypt image using AES-CBC."""
        block = self.process_file(input_path)
        pad = self.get_padding(block)
        block_trimmed = block[64:pad]
        plaintext = cipher.decrypt(block_trimmed)
        plaintext = block[0:64] + plaintext + block[pad:]
        self.save_file(output_path, plaintext)

    def calculate_npcr(self, image1_path: str, image2_path: str) -> float:
        """Calculate NPCR between two images."""
        img1 = np.array(Image.open(image1_path))
        img2 = np.array(Image.open(image2_path))
        diff = np.sum(img1 != img2)
        total_pixels = img1.shape[0] * img1.shape[1] * img1.shape[2]
        npcr = (diff / total_pixels) * 100
        return npcr

    def calculate_uaci(self, image1_path: str, image2_path: str) -> float:
        """Calculate UACI between two images."""
        img1 = np.array(Image.open(image1_path), dtype=np.float32)
        img2 = np.array(Image.open(image2_path), dtype=np.float32)
        diff = np.abs(img1 - img2)
        uaci = np.sum(diff) / (img1.shape[0] * img1.shape[1] * img1.shape[2] * 255) * 100
        return uaci

    def create_modified_image(self, image_path: str, output_path: str) -> None:
        """Create a modified version of the image."""
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.uint8)
        # Modify one pixel (flip the top-left corner)
        img_array[0, 0, 0] = np.uint8(int(img_array[0, 0, 0] + 1) % 256)  # R channel
        img_array[0, 0, 1] = np.uint8(int(img_array[0, 0, 1] + 1) % 256)  # G channel
        img_array[0, 0, 2] = np.uint8(int(img_array[0, 0, 2] + 1) % 256)  # B channel

        # Save the modified image
        modified_img = Image.fromarray(img_array)
        modified_img.save(output_path)

    def plot_correlation_analysis(self, image_path: str, title: str, samples: int = 5000) -> dict:
        """Plot and calculate pixel correlation in all directions."""
        img = np.array(Image.open(image_path))
        correlations = {}
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        plt.rcParams.update({'font.size': 12})
        directions = ['Horizontal', 'Vertical', 'Diagonal']
        
        for channel in range(3):
            channel_data = img[:, :, channel]
            height, width = channel_data.shape
            
            for direction_idx, direction in enumerate(directions):
                x_samples = []
                y_samples = []
                
                while len(x_samples) < samples:
                    i = np.random.randint(0, height - 1)
                    j = np.random.randint(0, width - 1)
                    
                    if direction == 'Horizontal':
                        if j < width - 1:
                            x_samples.append(channel_data[i, j])
                            y_samples.append(channel_data[i, j + 1])
                    elif direction == 'Vertical':
                        if i < height - 1:
                            x_samples.append(channel_data[i, j])
                            y_samples.append(channel_data[i + 1, j])
                    else:  # Diagonal
                        if i < height - 1 and j < width - 1:
                            x_samples.append(channel_data[i, j])
                            y_samples.append(channel_data[i + 1, j + 1])
                
                correlation = np.corrcoef(x_samples, y_samples)[0, 1]
                correlations[f"{['R', 'G', 'B'][channel]}_{direction}"] = correlation
                
                ax = axes[channel, direction_idx]
                ax.scatter(x_samples, y_samples, c=['red', 'green', 'blue'][channel], 
                          alpha=0.1, s=1)
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Adjacent Pixel Value")
                ax.set_title(f"{['R', 'G', 'B'][channel]} {direction}\nr = {correlation:.4f}")
                
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        self.save_plot(f"correlation_{title.lower().replace(' ', '_')}.png")
        
        return correlations
    
    def plot_rgb_histograms(self, image_path: str, title: str) -> None:
        """Plot RGB channel histograms."""
        img = np.array(Image.open(image_path))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.rcParams.update({'font.size': 12})
        colors = ['red', 'green', 'blue']

        for idx, (ax, color) in enumerate(zip(axes, colors)):
            hist, bins = np.histogram(img[:, :, idx].ravel(), bins=256, range=(0, 256))
            ax.bar(bins[:-1], hist, color=color, alpha=0.7, width=1)
            ax.set_title(f"{color.capitalize()} Channel")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        self.save_plot(f"histogram_{title.lower().replace(' ', '_')}.png")
    
    def calculate_entropy(self, image_path: str) -> dict[str, float]:
        """Calculate entropy for each RGB channel."""
        img = np.array(Image.open(image_path))
        entropy = {}
        
        for channel, color in enumerate(['R', 'G', 'B']):
            channel_data = img[:, :, channel].flatten()
            hist = np.histogram(channel_data, bins=256, range=(0, 256))[0]
            prob = hist / np.sum(hist)
            entropy[color] = -np.sum(prob * np.log2(prob + 1e-10))
            
        return entropy
    
    def calculate_psnr(self,image1_path, image2_path):
        # Load the images
        img1 = np.array(Image.open(image1_path), dtype=np.float32)
        img2 = np.array(Image.open(image2_path), dtype=np.float32)
        
        # Check if dimensions match
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for PSNR calculation.")
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((img1 - img2) ** 2)
        
        # Handle special case where MSE is zero (identical images)
        if mse == 0:
            return float('inf')  # Infinite PSNR (perfect match)
        
        # Calculate PSNR
        max_pixel_value = 255.0  # For 8-bit images
        psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
        
        return psnr


def main():
    # Initialize the analyzer
    analyzer = ImageEncryptionAnalyzer()

    # Find the most recent results directory
    results_dir =  analyzer.results_dir
    results_dir = max(glob.glob("results_*/images"), key=os.path.getctime)

    print("Starting image encryption analysis...")

    # Convert PNG to BMP if needed
    if not os.path.exists("baboon.bmp"):
        Image.open("baboon.png").save("baboon.bmp", format="BMP")

    # Construct file paths
    original_image_path = "baboon.bmp"
    encrypted_image_path = os.path.join(results_dir, "e_baboon_cbc.bmp")
    modified_image_path = os.path.join(results_dir, "baboon_modified.bmp")
    encrypted_modified_image_path = os.path.join(results_dir, "e_baboon_modified_cbc.bmp")

    # Process images
    # Initialize cipher
    cipher = analyzer.init_cipher()

    # Encrypt original image
    
    print("Encrypting original image...")
    analyzer.encrypt_image(cipher, original_image_path, encrypted_image_path)


    print("Creating modified image...")
    analyzer.create_modified_image(original_image_path, modified_image_path)

    
    print("Encrypting modified image...")
    analyzer.encrypt_image(cipher, modified_image_path, encrypted_modified_image_path)
    

    #Analysis
    print("\nAnalyzing original image...")
    original_corr = analyzer.plot_correlation_analysis("baboon.bmp", 
                                                     "Original Image Correlation")
    analyzer.plot_rgb_histograms("baboon.bmp", "Original Image Histograms")

    print("\nAnalyzing encrypted image...")
    encrypted_corr = analyzer.plot_correlation_analysis(
        encrypted_image_path,
        "Encrypted Image Correlation"
    )
    analyzer.plot_rgb_histograms(
        encrypted_image_path,
        "Encrypted Image Histograms"
    )
    # Add merged histogram plots
    print("\nGenerating merged histograms...")
    analyzer.plot_rgb_histograms("baboon.bmp", "Original Image")
    analyzer.plot_rgb_histograms(
        encrypted_image_path,
        "Encrypted Image"
    )

    # Calculate NPCR and UACI
    npcr_value = analyzer.calculate_npcr(encrypted_image_path, encrypted_modified_image_path)
    uaci_value = analyzer.calculate_uaci(encrypted_image_path, encrypted_modified_image_path)
    
    psnr = analyzer.calculate_psnr(original_image_path, encrypted_image_path)

    # Calculate entropy
    original_entropy = analyzer.calculate_entropy("baboon.bmp")
    encrypted_entropy = analyzer.calculate_entropy(
        encrypted_modified_image_path
    )
    
    # Save results to file
    with open(os.path.join(analyzer.results_dir, "analysis_results.txt"), "w") as f:
        f.write("Image Encryption Analysis Results\n")
        f.write("================================\n\n")
        
        f.write("1. Differential Analysis\n")
        f.write(f"NPCR: {npcr_value:.4f}% (Expected: >99.5%)\n")
        f.write(f"UACI: {uaci_value:.4f}% (Expected: ~33.4633%)\n\n")
        f.write(f"PSNR: {psnr:.4f} dB\n\n")

        f.write("2. Original Image Correlation\n")
        for key, value in original_corr.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("3. Encrypted Image Correlation\n")
        for key, value in encrypted_corr.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("4. Original Image Entropy\n")
        for channel, value in original_entropy.items():
            f.write(f"{channel}: {value:.4f}\n")
        f.write("\n")
        
        f.write("5. Encrypted Image Entropy\n")
        for channel, value in encrypted_entropy.items():
            f.write(f"{channel}: {value:.4f}\n")
            
    print(f"\nAnalysis complete! Results saved in {analyzer.results_dir}/")

if __name__ == "__main__":
    main()