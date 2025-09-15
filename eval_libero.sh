source examples/libero/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

TASK_SUITE=$1

python examples/libero/main.py --args.task_suite_name=${TASK_SUITE}