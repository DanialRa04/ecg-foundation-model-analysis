"""Read selected tensors from a trusted PyTorch ZIP without optimizer storage."""
import io
import pickle
import sys
import zipfile
import torch

def _tensor(storage, offset, size, stride, *unused):
    return {'storage': storage, 'offset': int(offset), 'size': tuple(size), 'stride': tuple(stride)}

def _parameter(data, *unused):
    return data

class _StructureReader(pickle.Unpickler):
    """Used only after the caller explicitly enables trusted checkpoint loading."""
    def persistent_load(self, value):
        if not value or value[0] != 'storage': raise pickle.UnpicklingError('Unsupported checkpoint storage')
        _, storage_type, key, location, count, *unused = value
        dtypes = {'FloatStorage':torch.float32,'HalfStorage':torch.float16,'DoubleStorage':torch.float64,'BFloat16Storage':torch.bfloat16,'LongStorage':torch.int64,'IntStorage':torch.int32,'ShortStorage':torch.int16,'ByteStorage':torch.uint8,'CharStorage':torch.int8,'BoolStorage':torch.bool,'ComplexFloatStorage':torch.complex64,'ComplexDoubleStorage':torch.complex128}
        if storage_type.__name__ not in dtypes: raise ValueError('Unsupported tensor storage dtype')
        return {'dtype':dtypes[storage_type.__name__], 'key':str(key), 'count':int(count)}

    def find_class(self, module, name):
        if module == 'torch._utils' and name in {'_rebuild_tensor','_rebuild_tensor_v2','_rebuild_tensor_v3'}: return _tensor
        if module == 'torch._utils' and name in {'_rebuild_parameter','_rebuild_parameter_with_state'}: return _parameter
        return super().find_class(module, name)

def open_structure(path, trusted=False):
    if not trusted: raise ValueError('Streaming author checkpoint metadata requires --trust-checkpoint')
    archive = zipfile.ZipFile(path)
    try:
        names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(names) != 1: raise ValueError('Expected one checkpoint metadata record')
        prefix = names[0].rsplit('/',1)[0]
        if f'{prefix}/byteorder' in archive.namelist() and archive.read(f'{prefix}/byteorder').decode() != sys.byteorder:
            raise ValueError('Checkpoint byte order differs from this host')
        data = _StructureReader(io.BytesIO(archive.read(names[0]))).load()
        if not isinstance(data, dict) or 'cfg' not in data or 'model' not in data: raise ValueError('Checkpoint requires cfg and model entries')
        return archive, prefix, data['cfg'], data['model']
    except Exception:
        archive.close(); raise

def read_tensor(archive, prefix, descriptor):
    storage = descriptor['storage']
    size, stride, offset = descriptor['size'], descriptor['stride'], descriptor['offset']
    if len(size) != len(stride) or offset < 0 or any(value < 0 for value in (*size,*stride)):
        raise ValueError('Invalid tensor view')
    if any(value == 0 for value in size): return torch.empty(size, dtype=storage['dtype'])
    last = offset + sum((length-1)*step for length,step in zip(size,stride))
    if last >= storage['count']: raise ValueError('Tensor view exceeds stored bounds')
    expected = storage['count'] * torch.empty((),dtype=storage['dtype']).element_size()
    entry = f"{prefix}/data/{storage['key']}"
    if archive.getinfo(entry).file_size != expected: raise ValueError('Checkpoint tensor byte count mismatch')
    # Read only the needed storage span into one CPU allocation, avoiding a
    # second full-sized bytes/bytearray copy for large checkpoint tensors.
    element_size = torch.empty((),dtype=storage['dtype']).element_size()
    values = torch.empty(last-offset+1,dtype=storage['dtype'])
    target = memoryview(values.view(torch.uint8).numpy()).cast('B')
    with archive.open(entry) as stream:
        stream.seek(offset * element_size)
        position = 0
        while position < len(target):
            count = stream.readinto(target[position:position+1024*1024])
            if not count: raise ValueError('Truncated checkpoint tensor')
            position += count
    return torch.as_strided(values,size=size,stride=stride,storage_offset=0)
