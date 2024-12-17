# from models.architectures import *

# model = Res2Net50(input_shape=(640, 640, 3), include_top=False, weights=None)
# model.summary()

print('runn')
import importlib
import os
from inspect import getmembers, isfunction
from contextlib import redirect_stdout

from models import architectures as arc
from models.architectures import *


mod = importlib.import_module(__name__)

for fn in getmembers(arc, isfunction):
    try:
        summa_file = os.path.join('./results', f'{fn[0]}_summary.txt')
        if not os.path.isfile(summa_file):
            model = getattr(mod, fn[0])(input_shape=(640, 640, 3), include_top=True, weights=None)
            with open(summa_file, 'w') as f:
                with redirect_stdout(f):
                    model.summary()
            
            del model
    except BaseException as e:
        print(e, fn[0])