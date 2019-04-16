import dataLogger
import random

logger = dataLogger.Logger()
for i in range(100):
    logger.log(i*random.randint(1,2))
logger.plot(file = 'testing')
