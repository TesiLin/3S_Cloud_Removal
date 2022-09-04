from importlib.util import find_spec
import os

if __name__ == '__main__':
    origin_spec = os.path.join(find_spec('attrdict').origin, '..')
    for file_name  in ['mapping.py', 'mixins.py', 'default.py', 'merge.py']:
        target = os.path.realpath(os.path.join(origin_spec, file_name))
        with open(target, 'r') as f:
            content = f.read()
        content = content.replace('from collections ', 'from collections.abc ')
        with open(target, 'w') as f:
            f.write(content)
    import attrdict.mapping
    print('patch success!')

        

