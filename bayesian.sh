#!/bin/bash
#SBATCH --job-name=job        # create a short name for your job
#SBATCH -p cryoem
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mj7341@princeton.edu

module purge
module load anaconda3/2023.3

conda activate cryodrgn_33

# python src/train_bayesian_logreg.py S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 -w 512 -s 512 -b 128 --epochs 20 -o s2-11_w512_bayes
# python src/train_bayesian_logreg.py S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 -w 512 -s 128 -b 128 --epochs 20 -o s2-11_w512_bayes

python src/eval_bayesian_logreg.py /scratch/gpfs/ZHONGE/mj7341/cos513/cos513-wesad/s2-11_w512_bayes/bayeslogreg_epoch20.pt \
    S13 S14 S15 S16 S17 \
    -o /scratch/gpfs/ZHONGE/mj7341/cos513/cos513-wesad/s2-11_w512_bayes/eval_s13-17 \
    -w 512 --stride 512 \
    -b 64
