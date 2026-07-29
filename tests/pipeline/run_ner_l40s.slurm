#!/bin/bash
#SBATCH --job-name=slm-ner-l40s
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=224G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-ner-l40s-%j.out
#
# TASK TYPE: NER (entity-span F1 scorer). Biomedical NER exercises the entity-diversity
# quality controls, the NER hard-negative path (harder-context, correct-type), and the
# span-F1 eval — a path the classification run never touches.
export SLM_CUDA_ISOLATION=1
export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
source /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/_l40s_task_body.sh
