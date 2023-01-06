# use sh to run this!
echo "Job being run!"
echo $1

#echo $2
#if [ $2 = "rq" ]; then
#  PARTITION="gpu_requeue"
#else 
#  PARTITION="gpu"
#fi
#echo "Using partition:"
#echo $PARTITION

# getting the number of experiments present
NUM_JOBS=`/home/tbb16/anaconda3/envs/core/bin/python $1.py`
#as it is zero indexed. 
echo "Number of Jobs!"
echo $NUM_JOBS

sed -i "s/#SBATCH --array.*/#SBATCH --array=0-$((NUM_JOBS-1))/" srun.sh
#sed -i "s/#SBATCH -p.*/#SBATCH -p $PARTITION/" srun.sh

sbatch --export=ALL,JOBS=$1 srun.sh 