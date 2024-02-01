import os

def get_extra_dependencies(package, module_name):
    file_path = os.path.join('requirements', package, f'requirements-{module_name}.txt')
    with open(file_path) as f:
        return f.read().splitlines()

# Extra dependencies package mapping
EXTRA_DEPENDENCIES_MAPPING = {
    'mysql': 'sql',
    'postgresql': 'sql',
    'sqlite': 'sql',
    'test': 'test',
    'chromadb': 'vectordb',
    'faiss': 'vectordb',
    'weaviate': 'vectordb',
}