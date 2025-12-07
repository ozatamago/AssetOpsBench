#!/bin/bash
# Activate conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

which python
python --version
python -m pip show agent_hive
python -m pip show reactxen
python -m pip show fmsr_agent
python -m pip show iotagent
python -m pip show tsfmagent

# Run the entire thing
python /home/run_track_1.py --utterance_ids "1,2,3,4,5,6,7,8,9,10,11,12,41,42,43,44,45,46,47,48"

# Keep the container alive
tail -f /dev/null
