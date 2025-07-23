#!/usr/bin/bash

# run this script with the following command:
# torchrun --no-python --nnodes 1 --nproc-per-node 2 ./run.sh

# Having run.sh file like this is preferred to running hello.py directly,
# because it allows us to add setup / teardown commands before the main python
# code is executed.

#!/usr/bin/bash
# edited run.sh script to run mount.sh
if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    echo "Mounting conda environment"
    source /packages/torchx_conda_mount/mount.sh
    echo "Done mounting conda environment"
fi

# gets the directory containing this script, cd, then runs hello.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR &&
python3 "$@"
