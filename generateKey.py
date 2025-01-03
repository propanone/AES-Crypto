"""
The
key we select should not be guessable and should provide sufficient entropy,
which simply means that the key should lack order or predictability. The fol-
lowing Python code will create a random key that is 16 bytes:
"""
import os
import binascii
import random
from Crypto.Cipher import AES
key = binascii.hexlify(os.urandom(16))
print ('key', [x for x in key] )

iv = ''.join([chr(random.randint(0, 0xFF)) for i in range(16)]) 

aes = AES.new(key, AES.MODE_CBC, iv)
data = 'Playing with AES' # <- 16 bytes
encd = aes.encrypt(data)