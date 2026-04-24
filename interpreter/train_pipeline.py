import time
import samples
import train_module
import data_generator


data_generator.generate()
start = time.time()
last_val_loss = train_module.train()
end = time.time()
computation_percentage = samples.run()
duration_minutes = (end - start) / 60
print(f'It took: {round(duration_minutes, 1)} min')
print(f'{last_val_loss=}')
print(f'time-complexity score: {1 / (last_val_loss * duration_minutes)}')
print(f'time-complexity computation score: { computation_percentage / duration_minutes }')
