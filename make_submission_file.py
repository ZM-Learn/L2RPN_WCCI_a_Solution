
from check_your_submission import main as test_submission
import time

model_dir = 'example_submissions/submission' # your submission directory

time_start=time.time()

test_submission(model_dir)

time_end=time.time()
print('time cost',time_end-time_start,'s')
