
import os

class Logger(object):
    def __init__(self,path):
        # 追加读写，文件不存在创建新文件
        self.file=open(path,'a+')

    def write(self,log):
        self.file.write(log)
        print(log)

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()
