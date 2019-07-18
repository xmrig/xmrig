#!/bin/bash

machine="$1"
max_t_cost="$2"
max_m_cost="$3"
max_lanes="$4"
branch="$5"
duration="$6"
queue="$7"
run_tests="$8"

if [ -z "$machine" ]; then
    echo "ERROR: Machine must be specified!" 1>&2
    exit 1
fi

if [ -z "$max_t_cost" ]; then
    max_t_cost=16
fi

if [ -z "$max_m_cost" ]; then
    max_m_cost=$((8 * 1024 * 1024))
fi

if [ -z "$max_lanes" ]; then
    max_lanes=16
fi

if [ -z "$branch" ]; then
    branch='master'
fi

if [ -z "$duration" ]; then
    duration=2h
fi

REPO_URL='https://github.com/WOnder93/argon2.git'

dest_dir="$(pwd)"

task_file="$(mktemp)"

cat >$task_file <<EOF
#!/bin/bash
#PBS -N argon2-cpu-$machine-$branch
#PBS -l walltime=$duration
#PBS -l nodes=1:ppn=$max_lanes:cl_$machine
#PBS -l mem=$(($max_m_cost / (1024 * 1024) + 1))gb
$(if [ -n "$queue" ]; then echo "#PBS -q $queue"; fi)

module add cmake-3.6.1

mkdir -p "$dest_dir/\$PBS_JOBID" || exit 1

cd "$dest_dir/\$PBS_JOBID" || exit 1

git clone "$REPO_URL" argon2 || exit 1

cd argon2 || exit 1

git checkout "$branch" || exit 1

(autoreconf -i && ./configure && make) || exit 1

if [ "$run_tests" == "yes" ]; then
    make check
fi

bash scripts/run-benchmark.sh $max_t_cost $max_m_cost $max_lanes \
    >"$dest_dir/\$PBS_JOBID/benchmark-$machine-$branch.csv"
EOF

qsub "$task_file"

rm -f "$task_file"
