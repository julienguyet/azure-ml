import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def build_model(data: pd.DataFrame) -> dict:
    X = data.drop(columns='Churn')
    y = data['Churn']
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=30)

    log_reg = LogisticRegression(solver='lbfgs',  max_iter=1000)
    log_reg.fit(X_train, y_train)
    
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    model_performance = {'accuracy': accuracy, 'classification_report': report}
    return model_performance