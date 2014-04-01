

import os
import tempfile

if 'EXPERIMENTS_FOLDER' in os.environ:
    experiments_folder = os.environ['EXPERIMENTS_FOLDER']
else:
    experiments_folder = os.path.join( tempfile.gettempdir(), 'spearmint_salad_experiments' )


print 'experimetns_folder: ', experiments_folder
if not os.path.exists(experiments_folder):
    os.makedirs(experiments_folder)
    
    
    
    