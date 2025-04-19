import datetime as dt
os.environ["CUDA_VISIBLE_DEVICES"]="3"
configfile: 'config.yml'
name = config['name']

rule preprocess_data:
    input: 
        file ='usa_0_regional_monthly.csv',
    output: 
        f'{name}_dict.pkl'
    script: "preprocessing.py" 

rule hypertune_kan:
    input:
        dataset = f'{name}_dict.pkl',
    output: 
        params = f'hyperparameters/{name}/kan_params.txt',
        results = f'hyperparameters/{name}/kan_R2.txt',
    script: "kan_hypertuning.py"

rule hypertune_fnn:
    input:
        dataset = f'{name}_dict.pkl',
    output: 
        params = f'hyperparameters/{name}/fnn_params.txt',
        results = f'hyperparameters/{name}/fnn_MSE.txt'
    script: "fnn_hypertuning.py"

# rule hypersort_kan:
#     input:
#         params = 'hyperparameters/{input.run_name}/kan_params.txt',
#         results = 'hyperparameters/{input.run_name}/kan_R2.txt',
#     output:
#         f'{name}_kan_best_params.pkl'
#     script: "kan_hypersort.py"

# rule hypersort_fnn:
#     input:
#         params = 'hyperparameters/{input.run_name}/fnn_params.txt',
#         results = 'hyperparameters/{input.run_name}/fnn_R2.txt',
#     output:
#         f'{name}_fnn_best_params.pkl'
#     script: "fnn_hypersort.py"

# rule run_kan:
#     input:
#         dataset = f'{name}_dict.pkl',
#         best_params = f'{name}_kan_best_params.pkl'
#     output:
#         spline_metrics = f'results/{name}_spline.csv'
#         symbolic_metrics = f'results/{name}_symbolic.csv'
#         equation_txt = f'equations/{name}.txt'
#         equation_ltx = 
#     script: "main.py"

# rule run_fnn:
#     input:
#         dataset = f'{name}_dict.pkl',
#         best_params = f'{name}_fnn_best_params.pkl'
#     output:
#         metrics = f'results/{name}_fnn.csv'
#         model = f'models/{name}_fnn.pt'

# rule get_shap_kan:
#     input:
#         equation = f'equations/{name}.txt'
#         dataset = f'{name}_dict.pkl'
#     output:
#         shap_values = f'shap-values/{save_as}_kan.pkl'
# rule get_shap_fnn:
#     input:
#         model = f'models/{name}_fnn.pt'
#         dataset = f'{name}_dict.pkl'
#     output:
#         shap_values = f'shap-values/{save_as}_fnn.pkl'
#     script:

# rule plot_shap:
#     input:
#         kan_values = 
#         fnn_values = 
#     output:
#         kan_plot = 
#         fnn_plot = 


    

