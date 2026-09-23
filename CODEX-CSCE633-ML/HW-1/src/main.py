import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List
from Sourabh_Pandit_code import *

def plot_iteration_loss(losses: list, plot_name: str, model_type: str) -> None:

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, marker='o', markersize=2, linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration" + f"\n({model_type})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name +".png", dpi=300, bbox_inches='tight')
    #plt.show()

def grid_search_logregr(X_train_scaled, y_train, X_val_scaled, y_val) -> None:

    print("\n\nHyperparameter tuning for Logistic Regression\n")
    eval_list= []

    for lr in [0.1, 0.05, 0.08, 0.0001]:
            for prob in [0.1, 0.2]:
                    for iter in [500, 1000, 2500, 5000]:
                        print(f"[LR: {lr:6.4f}, p_thresh: {prob:6.4f}, iter: {iter:7d}]",end='->')
                        log_model = LogisticRegression(learning_rate=lr,
                                                       max_iter=iter,
                                                       prob_threshold = prob)
                        y_binary = log_model.label_binarize(y_train)
                        y_val = log_model.label_binarize(y_val)

                        log_model.fit(X_train_scaled, y_binary)

                        y_pred = log_model.predict(X_train_scaled)
                        f1_score = log_model.F1_score(y_binary, y_pred)

                        y_pred_probs = log_model.predict_proba(X_train_scaled)
                        auroc = log_model.get_auroc(y_binary, y_pred_probs)

                        print (f"f1_score = {f1_score:6.4f}", end=' -> ')
                        print (f"auroc= {auroc:6.4f}")

                        eval_list.append((lr, prob, iter, f1_score, auroc))


                        if f1_score > 0.90 and auroc > 0.90:
                            print(f"\t{eval_list[-1]}: BREAKING")

                    print("")

def grid_search_linreg() -> None:

    print("\n\nHyperparameter tuning for Linear Regression\n")
    dp = DataProcessor("./")
    (df_train, df_test) = dp.load_data()

    for dframe in (df_train, df_test):
        if dp.check_missing_values(dframe) > 0:
            dp.clean_data(df_train)

    # ========================================= #
    # Train and Test Data
    # ========================================= #
    (X_raw, y_raw) = dp.extract_features_labels(df_train)
    (X_train, X_val, y_train, y_val) = dp.train_val_split(X_raw, y_raw)
    X_test = df_test.iloc[:, ].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    #X_train_scaled, min_vals, max_vals = dp.normalize(X_train)
    #X_val_scaled, _, _ = dp.normalize(X_val, min_vals, max_vals)
    #X_test_scaled, _, _ = dp.normalize(X_test, min_vals, max_vals)

    '''
    # alpha is the regularization strength
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.001):
        for iter in  (1000, 5000, 10000):
            lasso = Lasso(alpha=0.1, max_iter=10000)

            # Fit the model
            lasso.fit(X_train_scaled, y_train)
            y_pred = lasso.predict(X_val_scaled)

            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)

            print(f"RMSE: {rmse:.4f}")
            print(f"Model Coefficients: {lasso.coef_}")

    exit(0)

    (X_train, y_train) = dp.extract_features_labels(df_train)
    X_test = df_test.iloc[:, ].to_numpy()

    X_train_scaled, min_vals, max_vals = dp.normalize(X_train)
    X_test_scaled, _, _ = dp.normalize(X_test, min_vals, max_vals)
    '''

    #for iter in (1000, 5000, 10000, 50000):
    #    for lr in (0.1, 0.08, 0.05, 0.01, 0.008, 0.004, 0.001):
    #        for l2 in (0.1, 0.08, 0.05, 0.02, 0.01, 0.008, 0.005, 0.002, 0.001, 0.0008, 0.0004, 0.0002, 0.0001):

    #for lr in (0.1, 0.2, 0.3, 0.1):
    for lr in (0.05, 0.001, 0.01, 0.1):
        for iter in (15000, 10000, 5000, 2500, 500):
            for lambda_val in (0.005, 0.001, 0.05, 0.01, 0.5, 0.1):
                linear_model = LinearRegression(learning_rate = lr,
                                                max_iter = iter,
                                                l1_lambda=lambda_val,
                                                l2_lambda=0.0,
                                                reg_flag = 1)
                linear_model.fit(X_train_scaled, y_train)

                y_pred = linear_model.predict(X_val_scaled)

                rmse = linear_model.metric(y_val, y_pred)

                plot_name =  "lin_regr_loss_"                 + \
                            f"_iters_{linear_model.max_iter}" + \
                            f"_LR_{linear_model.learning_rate}"

                if rmse < 73.0:
                    print(f"[LR: {lr:6.4f}, max_iter: {iter:7d} l2_lambda: {l2:6.4f}]", end=' --> ')
                print(f"[LR: {lr:6.4f}, max_iter: {iter:7d} lambda: {lambda_val:6.4f}]", end=' --> ')
                print("RMSE:", rmse)

                y_test_pred = linear_model.predict(X_test_scaled)
        print("")


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Any, Callable, List, Tuple
from sklearn.linear_model import Lasso
import pandas as pd

def cross_validate_model(X: np.ndarray,
                         y: np.ndarray,
                         model_fn: Callable[[], Any],
                         n_splits: int = 5,
                         scale: bool = True,
                         random_state: int = 42) -> Tuple[List[float], float]:
    """
    Perform K-Fold cross-validation for a regression model.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        model_fn: A function that returns a fresh instance of the regression model (e.g., lambda: Lasso(alpha=0.1))
        n_splits: Number of folds for K-Fold cross-validation
        scale: Whether to apply StandardScaler
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
            - List of RMSE scores from each fold
            - Average RMSE across folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Feature scaling
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Train model
        model = model_fn()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    avg_rmse = np.mean(rmse_scores)
    return rmse_scores, avg_rmse

def main():

    #grid_search_linreg()
    #grid_search_logregr(X_train_scaled, y_train, X_val_scaled, y_val)
    #exit(0)

    dp = DataProcessor("./")
    (df_train, df_test) = dp.load_data()

    for dframe in (df_train, df_test):
        if dp.check_missing_values(dframe) > 0:
            dp.clean_data(df_train)

    # ========================================= #
    # Train and Test Data
    # ========================================= #
    #(X_train, y_train) = dp.extract_features_labels(df_train)
    #X_test = df_test.iloc[:, ].to_numpy()

    #X_test_scaled, _, _ = dp.normalize(X_test, min_vals, max_vals)
    #X_train_scaled, min_vals, max_vals = dp.normalize(X_train)

    '''
    (X_raw, y_raw) = dp.extract_features_labels(df_train)
    (X_train, X_val, y_train, y_val) = dp.train_val_split(X_raw, y_raw)
    X_test = df_test.iloc[:, ].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    '''


    '''
    # Load data
    df = pd.read_csv("data_train_25s.csv")

    for col in df.columns:
        mask = df[col] != -200  # mask to exclude -200
        mean_val = df.loc[mask, col].mean()
        df.loc[df[col] == -200, col] = mean_val

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values


    # Run cross-validation
    for alpha in (0.001, 0.05, 0.01, 0.1, 0.5):
        for iter in (500, 1000, 5000, 10000, 20000):
            rmse_scores, avg_rmse =
                        cross_validate_model(X, y,
                                 model_fn=lambda: Ridge(alpha=alpha, max_iter = iter))
            #rmse_scores, avg_rmse =
            #         cross_validate_model(X, y,
            #         model_fn=lambda: Lasso(alpha=alpha, max_iter = iter))

            #print("Per-fold RMSEs:", rmse_scores)
            print("Average RMSE:", avg_rmse)
    '''

    # =================================================================== #
    # Data handling:
    # =================================================================== #

    # Load data, create data frames for train and test data
    # The data dir is the same as the source code
    dp = DataProcessor("./")
    (df_train, df_test) = dp.load_data()

    # Check for missing values. Drop the rows with missing values
    for dframe in (df_train, df_test):
        if dp.check_missing_values(dframe) > 0:
            dp.clean_data(dframe)

    # =================================================================== #
    # Train and Test Data
    # =================================================================== #

    '''
    # Extract covariates matric and target vector for training
    # and test data (In our case test data does not have target var.
    (X_train, y_train) = dp.extract_features_labels(df_train)

    X_test = df_test.iloc[:, ].to_numpy()

    # Normalize the train test data
    X_train_scaled, min_vals, max_vals = dp.normalize(X_train)
    X_test_scaled, _, _ = dp.normalize(X_test, min_vals, max_vals)
    '''
    (X_raw, y_raw) = dp.extract_features_labels(df_train)
    (X_train, X_val, y_train, y_val) = dp.train_val_split(X_raw, y_raw)
    X_test = df_test.iloc[:, ].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    #X_train_scaled, min_vals, max_vals = dp.normalize(X_train)
    #X_val_scaled, _, _ = dp.normalize(X_val, min_vals, max_vals)
    #X_test_scaled, _, _ = dp.normalize(X_test, min_vals, max_vals)




    # =================================================================== #
    # EDA
    # =================================================================== #

    # Histogram using the raw training data
    dp.draw_histogram(df_train.iloc[:, :11], "feature_histogram_raw.png")

    # Histogram using the normalized training data
    dp.draw_histogram(pd.DataFrame(X_train_scaled), "feature_histogram_scaled.png")

    # Scatter plot for feature PT08.S2(NMHC) V/s feature PTO8.S5(O3)
    dp.draw_scatter_plot(df_train, 2, 7)

    # Pearson correlation - draw feature heatmap
    dp.get_pearson_corr(df_train)


    # =================================================================== #
    # Linear Regression
    # =================================================================== #

    '''
    # The following function call was used to identify optimums values for
    # the hyperparameters. Once the values are noted, this function is not
    # being called anymore, but those hyperparam values are being used

    grid_search_linreg()

    # Update: Re-enabled at the end, with reduced params for demonstrating the process
    '''

    linear_model = LinearRegression()

    linear_model.fit(X_train_scaled, y_train)
    y_pred = linear_model.predict(X_val_scaled)
    print(f"Linear Regression: Y_VAL_PRED: {y_pred}\n\n")

    rmse = linear_model.metric(y_val, y_pred)
    print("\nLinear Regression -  RMSE:", rmse)

    plot_name =  "lin_regr_loss_"                 + \
                f"_iters_{linear_model.max_iter}" + \
                f"_LR_{linear_model.learning_rate}"

    plot_iteration_loss(linear_model.losses, plot_name, "Linear Regression")

    y_test_pred = linear_model.predict(X_test_scaled)
    print(f"Linear Regression: Y_TEST_PRED: {y_test_pred}\n\n")

    # =================================================================== #
    # Logistic Regression
    # =================================================================== #

    '''
    # The following function call was used to identify optimums values for
    # the hyperparameters. Once the values are noted, this function is not
    # being called anymore, but those hyperparam values are being used

    grid_search_logregr()

    # Update: Re-enabled at the end, with reduced params for demonstrating the process
    '''

    # =================================================================== #
    # Train and Test Data
    # =================================================================== #

    #log_model = LogisticRegression(learning_rate=0.0800,
    #                               max_iter=10000 ,
    #                               prob_threshold=0.3500)

    log_model = LogisticRegression()
    y_binary = log_model.label_binarize(y_train)

    log_model.fit(X_train_scaled, y_binary)

    y_pred = log_model.predict(X_train_scaled)
    f1_score = log_model.F1_score(y_binary, y_pred)

    y_pred_probs = log_model.predict_proba(X_train_scaled)
    auroc = log_model.get_auroc(y_binary, y_pred_probs)

    print(f"\nLogistic Regression - f1_score = {f1_score:6.4f}")
    print(f"Logistic Regression - AUROC= {auroc:6.4f}\n")

    plot_name =  "log_regr_loss_"                  + \
                f"_iters_{log_model.max_iter}"     + \
                f"_LR_{log_model.learning_rate}"

    plot_iteration_loss(log_model.losses, plot_name, model_type="Logistic Regression")

    print("\nNow predicting on the test data")
    y_pred = log_model.predict(X_test_scaled)
    y_pred_probs = log_model.predict_proba(X_test_scaled)

    print(f"Logistic Regression: Y_PRED_PROBS: {y_pred_probs}\n\n")

    # =================================================================== #
    # K-Fold Cross Validation
    # =================================================================== #
    evaluator = ModelEvaluator(n_splits= 5,  random_state = 42)

    # =================================================================== #
    # Cross Validation for Linear Regression Model
    # =================================================================== #
    linear_scores = evaluator.cross_validation(linear_model, X_train_scaled, y_train)
    print("\nCross Validation")
    print("\n   Cross-Validation Linear Regression: Avg RMSE:", np.mean(linear_scores))
    print("   Cross-Validation Linear Regression: Std Dev RMSE:", np.std(linear_scores), "\n\n\n")


    # =================================================================== #
    # Cross Validation for Logistic Regression Model
    # =================================================================== #
    logistic_scores = evaluator.cross_validation(log_model, X_train_scaled, y_binary)
    f1_scores, aurocs = zip(*logistic_scores)
    print(f"\n   Cross-Validation Log Regr: ", end=' ')
    print(f"        Avg F1: {np.mean(f1_scores):6.4f}, Std: {np.std(f1_scores):6.4f}")
    print(f"        Avg AUROC: {np.mean(aurocs):6.4f}, Std: {np.std(aurocs):6.4f}")

    # =================================================================== #
    # ROC Curve
    # =================================================================== #
    evaluator = ModelEvaluator()
    evaluator.plot_roc_per_fold(log_model, X_train_scaled, y_binary)

    # =================================================================== #
    # Hyperparameter Tuning Linear Regression
    # =================================================================== #
    #grid_search_linreg()

    # =================================================================== #
    # Hyperparameter Tuning Logistic Regression
    # =================================================================== #
    #grid_search_logregr()


if __name__ == "__main__":
    main()
    print("Hello World!")

