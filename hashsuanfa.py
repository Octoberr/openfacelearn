# coding:utf-8
import hashlib

md5 = hashlib.md5()
md5.update('my name is swm')
print md5.hexdigest()