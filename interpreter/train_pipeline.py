import time

from . import samples
from . import train_module
from . import data_generator
start = time.time()


data_generator.generate()
last_val_loss = train_module.train()

computation_percentage = samples.run()
duration_minutes = (time.time() - start) / 60
print(f'It took: {round(duration_minutes, 1)} min')
print(f'{last_val_loss=}')
print(f'time-complexity score: {1 / (last_val_loss * duration_minutes)}')
print(f'time-complexity computation score: { computation_percentage / duration_minutes }')
