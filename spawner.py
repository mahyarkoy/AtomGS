import subprocess
import sys
import argparse
import os

EXP_CONFIGS = {
    'atomgs_bicycle':
    {
        'cmd': 'python train.py',
        'save_dir_key': 'model_path',
        'model_path': 'atomgs/bicycle_full',
        'source_path': 'data/360_v2/bicycle',
        'split_path': 'None' # 'data/360_v2/bicycle/split_pca3_50top.json' 
    },

    'atomgs_flowers':
    {
        'cmd': 'python train.py',
        'save_dir_key': 'model_path',
        'model_path': 'atomgs/flowers_split_pca3_20top',
        'source_path': 'data/360_v2/flowers',
        'split_path': 'data/360_v2/flowers/split_pca3_20top.json' 
    },

    'atomgs_garden':
    {
        'cmd': 'python train.py',
        'save_dir_key': 'model_path',
        'model_path': 'atomgs/garden_split_pca3_20top',
        'source_path': 'data/360_v2/garden',
        'split_path': 'data/360_v2/garden/split_pca3_20top.json' 
    },

    'atomgs_stump':
    {
        'cmd': 'python train.py',
        'save_dir_key': 'model_path',
        'model_path': 'atomgs/stump_full',
        'source_path': 'data/360_v2/stump',
        'split_path': 'None' # 'data/360_v2/stump/split_pca3_20top.json' 
    },

    'atomgs_treehill':
    {
        'cmd': 'python train.py',
        'save_dir_key': 'model_path',
        'model_path': 'atomgs/treehill_split_pca3_20top',
        'source_path': 'data/360_v2/treehill',
        'split_path': 'data/360_v2/treehill/split_pca3_20top.json' 
    },
}

SBATCH_STR = '''#!/bin/bash 
#SBATCH --job-name={job_name} 
#SBATCH --ntasks=1 
#SBATCH --account=gard 
#SBATCH --qos=premium 
#SBATCH --partition=ALL 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:{device} 
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output={save_dir}/sbatch_{job_name}.out
#SBATCH --error={save_dir}/sbatch_{job_name}.out
#SBATCH --open-mode=truncate

echo "HOSTNAME: "$(hostname)
echo "tmpdir for the job: "$TMPDIR 
echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES 
echo "CPU allocated: "$(taskset -c -p $$)
echo "GPU allocated: "$CUDA_VISIBLE_DEVICES
nvidia-smi
source /nas/home/mkhayat/.bashrc
conda activate {condaenv}
'''

SBATCH_STR_LARGE = '''#!/bin/bash 
#SBATCH --job-name={job_name} 
#SBATCH --ntasks=1 
#SBATCH --account=gard 
#SBATCH --qos=premium_memory
#SBATCH --partition=large_gpu
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:{device}
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output={save_dir}/sbatch_{job_name}.out
#SBATCH --error={save_dir}/sbatch_{job_name}.out
#SBATCH --open-mode=truncate

echo "HOSTNAME: "$(hostname)
echo "tmpdir for the job: "$TMPDIR 
echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES 
echo "CPU allocated: "$(taskset -c -p $$)
echo "GPU allocated: "$CUDA_VISIBLE_DEVICES
nvidia-smi
source /nas/home/mkhayat/.bashrc
conda activate {condaenv}
'''

SBATCH_STR_ECLAIR = '''#!/bin/bash 
#SBATCH --job-name={job_name} 
#SBATCH --ntasks=1 
#SBATCH --account={partition}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --gpus={device}
#SBATCH --mem=32G
#SBATCH --output={save_dir}/sbatch_{job_name}.out
#SBATCH --error={save_dir}/sbatch_{job_name}.out
#SBATCH --open-mode=truncate

echo "HOSTNAME: "$(hostname)
echo "tmpdir for the job: "$TMPDIR 
echo "CPU allocated: "$(taskset -c -p $$)
echo "GPU allocated: "$CUDA_VISIBLE_DEVICES
nvidia-smi
source /nas/home/mkhayat/.bashrc
conda activate {condaenv}
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
echo $PYTORCH_CUDA_ALLOC_CONF
'''

COPY_STR = 'cp -r {temp_dir} {save_dir}'
DEL_STR = 'rm -r {temp_dir}'

def os_cmd(cmd, env=None, stdout=subprocess.PIPE):
    with subprocess.Popen(cmd.split(), stdout=stdout, stderr=subprocess.STDOUT,
        encoding='latin-1', env=env) as p:
        out, err = p.communicate()
    if err:
        print(f'\n>>> ERROR: The process raised an error: {err.decode()}\n')
    return out

def branch_dict(in_dict):
    '''
    For any list or tuple in in_dict, will branch into all combination.
    Returns a list of dict.
    '''
    collect = list()
    for key, val in in_dict.items():
        if len(collect) == 0:
            if isinstance(val, (list, tuple)):
                collect.extend([{key: sub_val} for sub_val in val])
            else:
                collect.append({key: val})
        else:
            new_collect = list()
            for cdict in collect:
                if isinstance(val, (list, tuple)):
                    for sub_val in val:
                        new_collect.append(dict(**cdict, **{key: sub_val}))
                else:
                    new_collect.append(dict(**cdict, **{key: val}))
            collect = new_collect
    return collect

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Root results directory.')
    parser.add_argument('--config', nargs='*', help=f'Choose from {list(EXP_CONFIGS.keys())}.')
    parser.add_argument('--single_thread', action='store_true', help='Serialize seeds on the same run.')
    parser.add_argument('--save_local', action='store_true', help='Saves on the local machine and copies back.')
    parser.add_argument('--bash', action='store_true', help='Runs with bash instead of sbatch.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--device', help='The gpu ids to use, e.g. 0,1,2. will be ignored when bash is not used.')
    parser.add_argument('--cluster', default='eclair', help='SBATCH option: use internal or eclair.')
    parser.add_argument('--partition', default='medium-lg', help='choice of partition for eclair: long-lg.')
    parser.add_argument('--env', help='Conda environment.')
    return parser.parse_args()

if __name__ == '__main__':
    args = setup_args()
    sbatch_str = SBATCH_STR_ECLAIR if args.cluster=='eclair' else SBATCH_STR
    for config_name in args.config:
        print(f'\n>>> Spawning {config_name}')
        config_base = EXP_CONFIGS[config_name]
        
        ### Branch config into a list of configs
        configs = branch_dict(config_base)

        cmd_list = list()
        job_names = list()
        job_save_dirs = list()
        ### Convert into command string
        for config in configs:
            ### Preprocess args
            for key, val in config.items():
                if isinstance(val, str):
                    config[key] = val.format(**config)
            save_dir_key = config['save_dir_key']
            del config['save_dir_key']
            config_save_dir = config[save_dir_key]
            temp_save_dir = (os.path.join('$TMPDIR/logs_temp', config_save_dir)
                    if args.save_local else os.path.join(args.save_dir, config_save_dir))
            config[save_dir_key] = temp_save_dir

            ### Make command string
            config_str = ' '.join([config['cmd']] + [f'--{key} {val}' for key, val in config.items() if key != 'cmd'])
            if args.save_local:
                config_str = f'{config_str}\n{COPY_STR.format(temp_dir=temp_save_dir, save_dir=args.save_dir)}\n{DEL_STR.format(temp_dir=temp_save_dir)}'
            cmd_list.append(config_str)
            job_names.append(config_save_dir.replace('/', '_'))
            job_save_dirs.append(os.path.join(args.save_dir, config_save_dir))
            os.makedirs(job_save_dirs[-1], exist_ok=True)

        ### Serialize the runs of the seeds if requires
        if args.single_thread:
            cmd_list = ['\n'.join(cmd_list)]
            job_names = f'{config_name}_single_thread'
            
        ### Run
        for cmd_str, job_name, job_save_dir in zip(cmd_list, job_names, job_save_dirs):
            if args.bash:
                ### Save sbatch to file
                bash_file_path = os.path.join(job_save_dir, f'bash_{job_name}.sh')
                set_device_str = f'export CUDA_VISIBLE_DEVICES={args.device}\n' if args.device is not None else ''
                with open(bash_file_path, 'w+') as fs:
                    print('#!/bin/bash\n'+set_device_str+'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512\n'+cmd_str, file=fs)
                    fs.flush()
                    os.fsync(fs)

                    ### Run the experiment
                    os_cmd(f'bash {bash_file_path}', stdout=sys.stdout)
            else:
                ### Save sbatch to file
                sbatch_cmd = sbatch_str.format(device=args.num_gpus, job_name=job_name, save_dir=job_save_dir, condaenv=args.env, partition=args.partition)
                sbatch_file = sbatch_cmd+'\n'+cmd_str
                sbatch_file_path = os.path.join(job_save_dir, f'sbatch_{job_name}.sh')
                with open(sbatch_file_path, 'w+') as fs:
                    print(sbatch_file, file=fs)
                    fs.flush()
                    os.fsync(fs)

                    ### Run the experiment
                    os_cmd(f'sbatch {sbatch_file_path}', stdout=sys.stdout)
