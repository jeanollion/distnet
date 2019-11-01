import os
# code from aparamon: https://github.com/h5py/h5py/issues/934
class AtomicFileHandler:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.pos = 0

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        else:
            self.pos = os.lseek(self.fd, pos, whence)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size):
        b = os.pread(self.fd, size, self.pos)
        self.pos += len(b)
        return b
