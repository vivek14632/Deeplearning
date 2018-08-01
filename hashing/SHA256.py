# Author: vivek
# Purpose: Library for SHA 256 conversion

import hashlib, binascii

def convertToSHA256(raw,key='salt'):
	dk =  hashlib.pbkdf2_hmac('sha256', raw.encode('utf-8'), key.encode('utf-8'), 100000)
	return binascii.hexlify(dk)


print((convertToSHA256('vivek')))


# Help tutorials
# https://www.youtube.com/watch?v=TkWAgeSYL_Q

