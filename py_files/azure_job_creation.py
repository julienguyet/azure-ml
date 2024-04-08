from azureml.core import * 

workspace = Workspace.from_config() # here we tell the instance which workspace to use

env = Environment.from_conda_specification(name='my_environment', file_path='../environment.yml') # this will tell Azure which dependencies to install so the job can run
experiment_name = 'predictions'
experiment = Experiment(workspace, experiment_name) 

script_run_config = ScriptRunConfig(source_directory='/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>',
                                    script='ModelPredict.py',
                                    arguments=['--dataset', '/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>'],
                                    environment=env)

run = experiment.submit(script_run_config)