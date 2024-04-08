# Deploy a Machine Learning Model on Azure

## 1. Go to Microsoft Azure Portal
To get started go to the Azure portal (see link below) and click on "*Try Azure for free*" on the right top corner of the page. 
This will ask you to create a (free) microsoft account and enjoy $200 credits on Azure.

Azure Portal Link: https://azure.microsoft.com/en-us/get-started/azure-portal

**Important Information***: please note you will still have to provide a credit card when creating your account. Make sure your email address is not already linked to a Microsoft account otherwise you will not enjoy the free credits from Azure and might be charged. 
Even though this tutorial is designed to limit fees, I would recommend deleting the Azure instance when you are done with your project. 

## 2. Create a Resource Group

Once your account is created, you will be redirected to the Azure home page. It will propose you different tutorials, feel free to explore the different modules relevant to you.
Below the welcome message, click on "*Resource Groups*" :

<img width="1112" alt="azure_home" src="https://github.com/julienguyet/azure-ml/assets/55974674/0df4e0b4-0af1-49a0-bb4e-f70660352f48">

Then, give your resource group an explicit name and select a region for your instance. For example, I called mine "*churn-prediction*" as my ML model tries to predict customer churn. 

<img width="736" alt="azure_resource_group" src="https://github.com/julienguyet/azure-ml/assets/55974674/6b158b4a-bfc7-4e7b-9892-54d6a4e7b1a7">

## 3. Set up the Machine Learning Workspace

The next step is to create a workspace and link it to the resource group we just initialized. 

To do this, go back to the home page and click on "*Azure Machine Learning*" and create a new workspace. We allocate this workspace to the resource group created before. Please note here we left networking as public so test can be realized easily, but only approved connections should be allowed in a real production environment.

<img width="379" alt="azure_workspace" src="https://github.com/julienguyet/azure-ml/assets/55974674/1304a1aa-e11e-467d-a919-80023d4a4c0b">

<img width="789" alt="azure_workspace-2" src="https://github.com/julienguyet/azure-ml/assets/55974674/68b74c4b-93d3-4904-9e6a-eb8872676a5d">

Once deployment is finished, we can go to the resource group and click on “Launch Studio”.  This will redirect us to the Azure AI | Machine Learning Studio where we can start building our model.

## 4. Machine Learning Studio

In there, in the “manage” section, configure a Compute Engine with a Standard_DS11_v2 virtual machine (tailored for simple python code / notebooks). We select the same for the compute cluster (with one node only as we will not need more computation power for this trial). 

Our environment is finally set up, we can now go to the “notebooks” section under “Authoring”.
You should be able to load local files to the instance from the dropdown menu. Clone the github onto your local machine and then manually load the folder to the instance.

<img width="410" alt="azure_notebook" src="https://github.com/julienguyet/azure-ml/assets/55974674/c1024912-fd6e-4fdb-b822-22914ba3a7ac">

Once upload is finished, open the Ecommerce-final notebook. At the top select your instance and Azure ML as kernel, and run the below code:

```
import pandas as pd
import sys
sys.path.append('/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>')

from py_files.train import build_model
from py_files.preprocess import preprocess

training_data_df = pd.read_csv('../data/Dataset/ECommerce.csv')

X_train = preprocess(training_data_df)

model_performance_dict = build_model(X_train)

print(model_performance_dict)
```

This code will load the python functions stored in the py_files folder and train the model. This will store both the model and its encoder to joblib files. This allow us to easily load the model during inference.
Do not forget to **update the paths** based on your set up (a console tab is available at the top of the notebook to help you exploring the instance).
After running the code you should get below output:

<img width="683" alt="training_code" src="https://github.com/julienguyet/azure-ml/assets/55974674/711039a2-d797-433f-9968-edab91e066d6">

Then, run the prediction code to make sure it works just fine:

```
import pandas as pd
from py_files.inference import make_predictions

user_data_df = pd.read_csv('../data/Dataset/test.csv')

# Call make_predictions function to generate predictions
predictions = make_predictions(user_data_df)

print(predictions)
```

Please note here we do not care about the performance of the model (we are using a simple Logistic Regression with little feature engineering), we are only interested in its deployment. 

<img width="672" alt="prediction_code" src="https://github.com/julienguyet/azure-ml/assets/55974674/15c6fc84-1d46-4fb4-b2e9-006dcd7d5975">

## 5. Prediction Job

Now that our model is operational, we need to create a job so the model can be deployed and accessed at any time. 
For example, let’s assume the marketing team has a file with a list of users and an application on which they can upload the file and ask for predictions, so they get the list of users flagged as churner and then run a marketing campaign to retain those users. The job would be triggered by an API linked to the instance and the app.

First, make sure the ModelPredict.py file has been stored to your instance:

<img width="323" alt="Screenshot 2024-04-07 at 18 24 35" src="https://github.com/julienguyet/azure-ml/assets/55974674/45fa5ff4-e126-4354-9684-2cce6f1e4925">

Then, execute the below code in the notebook:

```
import pandas as pd
from py_files.inference import make_predictions

user_data_df = pd.read_csv('../data/Dataset/test.csv')

# Call make_predictions function to generate predictions
predictions = make_predictions(user_data_df)

print(predictions)
```

Wait for job to be created, and then on the left menu, below "*Assets*", click on "*Jobs*". You should see this:

<img width="695" alt="azure-job" src="https://github.com/julienguyet/azure-ml/assets/55974674/783d368e-c2fa-4f77-a224-61f9a3c90c84">


When we click on it we can see it was executed successfully. We will now register the model:

<img width="612" alt="register_model" src="https://github.com/julienguyet/azure-ml/assets/55974674/6b1d0511-2e43-4278-954e-d0461bc49464">

Once registered, the model can be deployed:

<img width="612" alt="model_deploypment" src="https://github.com/julienguyet/azure-ml/assets/55974674/27a0bc93-e6c3-43a0-84a7-4c449004dc17">


Note that we will stop here as deploying the model can lead to some costs (allocated IP address, etc.). Finishing this deployment step would allow you to define endpoints, authentication method, job schedules and computation power allocated to this job.

Finally, here we have done this demonstration with always using the same file to avoid computation cost. In a real-life example, another script would be needed to upload new files submitted by user and make predictions on that file only.
