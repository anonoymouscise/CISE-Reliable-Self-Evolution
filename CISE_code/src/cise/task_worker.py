import sys
from .runtime import CONFIG

if __name__=='__main__':
    task=sys.argv[1]
    if task not in CONFIG['tasks']:raise ValueError('Task not configured')
    if CONFIG['method']=='cci':
        from .cci.runner import run
        run(task)
    else:
        from .llema.runner import generate
        generate(task)
