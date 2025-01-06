from cryptography.fernet import Fernet


def encrypt(filename, newfile, key):
    """
    Given a plain image (str), the new file name,  and key (bytes), it 
    encrypts the file and write it
    """
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read all file data
        file_data = file.read()
    # encrypt data
    encrypted_data = f.encrypt(file_data)
    # write the encrypted file
    with open(newfile, "wb") as file:
        file.write(encrypted_data)

        
def decrypt(filename, newfile, key):
    """
    Given a encrypted file (str), the new file name, and key (bytes), it 
    decrypts the file and write it
    """
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read the encrypted data
        encrypted_data = file.read()
    # decrypt data
    decrypted_data = f.decrypt(encrypted_data)
    # write the original file
    with open(newfile, "wb") as file:
        file.write(decrypted_data)


key = Fernet.generate_key()
enc = encrypt("baboon.png", "e_baboon.png", key)
dec = decrypt("e_baboon.png", "d_baboon.png", key)