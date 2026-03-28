import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''

class DataProcessor:
    def __init__(self, data_root: str):
        """Initialize data processor with paths to train and test data.

        Args:
            data_root: root path to data directory
        """
        self.data_root = data_root

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files.

        Returns:
            Tuple containing training and test dataframes
        """
        # TODO: Implement data loading

        df_train = pd.read_csv(self.data_root + "/" + "data_train_25s.csv")
        df_test = pd.read_csv(self.data_root  + "/" + "data_test_25s.csv")
        return (df_train, df_test)

    def check_missing_values(self, data: pd.DataFrame) -> int:
        """Count number of missing values in dataset.

        Args:
            data: Input dataframe

        Returns:
            Number of missing values
        """
        # TODO: Implement missing value check
        return data.isnull().sum().sum()

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.

        Args:
            data: Input dataframe

        Returns:
            Cleaned dataframe
        """
        # TODO: Implement data cleaning
        for col in data.columns:
            mask = data[col] != -200  # mask to exclude -200
            mean_val = data.loc[mask, col].mean()
            data.loc[data[col] == -200, col] = mean_val

        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.

        Args:
            data: Input dataframe

        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction

        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        return (X, y)

    def train_val_split(self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2,
                    seed: int = 1005) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Shuffle and split the dataset into training and validation sets.

        Args:
            X: Feature matrix
            y: label vector
            val_ratio: Proportion of data to use for validation
            seed: Random seed for reproducibility

        Returns: Tuple: (X_train, X_val, y_train, y_val)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_index = int((1 - val_ratio) * n_samples)
        train_idx, val_idx = indices[:split_index], indices[split_index:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        return X_train, X_val, y_train, y_val


    def get_pearson_corr(self, data: pd.DataFrame) -> None:
        """Return Pearson correlation coefficient for two features

        Args:
            data: Input dataframe
            features1_idx: index for features 1
            features2_idx: index for features 2

        Returns:
            returns nothing
        """

        # Pearson's correlation matrix
        corr_matrix = data.corr(method='pearson')
        plt.figure(figsize=(12, 10))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .75}
        )

        plt.title("Pearson Correlation Heatmap", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("feature_heatmap.png", dpi=300, bbox_inches='tight')
        #plt.show()

    def draw_histogram(self, data: pd.DataFrame, hist_name: str) -> None:
        """Draw histogram for all features and the target variable

        Args:
            data: Input dataframe

        Returns:
            returns Nothing
        """
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))  # 4 rows x 3 columns
        axes = axes.flatten()  # Flatten to 1D array for easy indexing

        for i, feature in enumerate(data.columns):
            axes[i].hist(data[feature], bins=20, edgecolor='black')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')

        if len(data.columns) < len(axes):
            axes[len(data.columns)].axis('off')

        plt.tight_layout()
        plt.savefig(hist_name, dpi=300, bbox_inches='tight')
        #plt.show()


    def draw_scatter_plot(self, data: pd.DataFrame, idx1: int, idx2: int) -> None:
        """Extract features and labels from dataframe, convert to numpy arrays.

        Args:
            data: Input dataframe
            features1_idx: index for features 1
            features2_idx: index for features 2

        Returns:
            returns Nothing
        """
        feature1 = data.columns[idx1]
        feature2 = data.columns[idx2]

        plt.figure(figsize=(8, 6))
        plt.scatter(data[feature1], data[feature2], alpha=0.6, edgecolors='k')

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Scatter Plot: {feature1} vs {feature2}')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')
        #plt.show()

    def normalize(self, X: np.ndarray,
                          min_vals: np.ndarray=None,
                          max_vals: np.ndarray=None) -> (np.ndarray, np.ndarray, np.ndarray):
        """
            Normalize each feature (column) in the NumPy array to the range [0, 1].

        Args:
            X: NumPy array of shape (n_samples, n_features)

        Returns:
            A new NumPy array with normalized values
        """
        X_normalized = X.copy().astype(float)  # Make a float copy for division

        if min_vals is None or max_vals is None:
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)

        for i in range(X.shape[1]):
            if min_vals[i] != max_vals[i]:
                X_normalized[:, i] = (X[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                X_normalized[:, i] = 0.0

        return X_normalized, min_vals, max_vals

class LinearRegression:
    def __init__(self,
                 learning_rate: float = 0.05,
                 max_iter: int = 1000,
                 l2_lambda: float = 0.005):
        """Initialize linear regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """

        self.weights = None # We do not know #features yet.
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_lambda = l2_lambda

        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training

        n_samples, n_features = X.shape

        if self.weights is None:
            self.weights = np.random.randn(n_features) * 0.01

        for _ in range(self.max_iter):
            y_pred = self.predict(X, scaled = True)
            error = y - y_pred

            # A float is an approx value. Anything less than 0.0001 if fluke
            if (self.l2_lambda < 0.0001):
                # No Regularization
                loss = self.criterion(y, y_pred)
                dw = -2  * X.T @ error / n_samples
            else:
                # Ridge loss = MSE + L2 penalty
                loss = self.criterion(y, y_pred) + self.l2_lambda * np.sum(self.weights ** 2)
                dw = (-2 * X.T @ error / n_samples) + 2 * self.l2_lambda * self.weights

            np.clip(dw, -1e5, 1e5, out=dw)

            db = -2 * np.sum(error) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.losses.append(loss)

        return self.losses

    def predict(self, X: np.ndarray, scaled:bool = False) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction

        return (X @ self.weights + self.bias)

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # TODO: Implement loss function
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation

        return np.sqrt(self.criterion(y_true, y_pred))

class LogisticRegression:
    def __init__(self,
                 learning_rate: float = 0.1,
                 max_iter: int = 2000,
                 prob_threshold: float=0.2):
        """Initialize logistic regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            prob_threshold: Probability Threshold to classify binary output as 1 or 0
        """
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.losses = []
        self.prob_threshold = prob_threshold


    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training

        n_samples, n_features = X.shape

        if self.weights is None:
            self.weights = np.random.randn(n_features) * 0.01

        for _ in range(1, self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            loss = self.criterion(y, y_pred)
            self.losses.append(loss)

            dw = np.dot(X.T, (y_pred -y)) / n_samples
            db = np.sum(y_pred -y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.

        Args:
            X: Feature matrix

        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction

        y_pred_proba = self.predict_proba(X)
        return np.where(y_pred_proba >= self.prob_threshold, 1, 0)

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # TODO: Implement loss function
        epsilon = 1e-10  # to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            F1 score (between 0 and 1)
        """

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * precision * recall / (precision + recall + 1e-15)

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.

        Args:
            y: Target vector

        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization
        return (y > 1000).astype(int)

    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities

        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation

        # Sort by predicted probability
        sorted_indices = np.argsort(-y_pred)
        y_true_sorted = y_true[sorted_indices]

        cum_pos = np.cumsum(y_true_sorted)
        total_pos = np.sum(y_true_sorted)
        total_neg = len(y_true_sorted) - total_pos

        tpr = cum_pos / (total_pos + 1e-15)
        fpr = np.cumsum(1 - y_true_sorted) / (total_neg + 1e-15)

        return np.clip(np.trapezoid(y=tpr, x=fpr), 0.0, 1.0)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation
        return self.get_auroc(y_true, y_pred)

    def sigmoid(self, z: float) -> float:
        """Calculate the value of the sigmoid function.

        Args:
            z: Paramt W-Transpose*X

        Returns:
            float value of the sigmoid function
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

class ModelEvaluator:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize evaluator with number of CV splits.

        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform cross-validation

        Args:
            model: Model to be evaluated
            X: Feature matrix
            y: Target vector

        Returns:
            List of metric scores
        """
        # TODO: Implement cross-validation

        scores = []

        for train_indices, val_indices in self.kf.split(X):
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)

            if hasattr(model, "F1_score") and hasattr(model, "get_auroc"):
                #  This is our Logistic Regression
                y_pred_probs = model.predict_proba(X_val_scaled)
                y_pred_labels = model.predict(X_val_scaled)

                f1 = model.F1_score(y_val, y_pred_labels)
                auroc = model.get_auroc(y_val, y_pred_probs)
                scores.append((f1, auroc))

            else:
                y_pred = model.predict(X_val_scaled)
                rmse = model.metric(y_val, y_pred)
                scores.append(rmse)

        return scores

    def plot_roc_per_fold(self, model, X_train_bin: np.ndarray, y_train_bin: np.ndarray) -> None:
        """
        Plot ROC curves and compute AUROC for each fold using Logistic Regression.

        Args:
            model: Logistic Regression model for which ROC curve is desired
            X_train_bin: Feature matrix
            y_train_bin: Target vector

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X_train_bin), 1):
            X_train, X_val = X_train_bin[train_idx], X_train_bin[val_idx]
            y_train, y_val = y_train_bin[train_idx], y_train_bin[val_idx]

            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_val)

            thresholds = np.sort(np.unique(y_probs))[::-1]
            tprs, fprs = [], []

            total_pos = np.sum(y_val)
            total_neg = len(y_val) - total_pos

            for thresh in thresholds:
                y_pred = (y_probs >= thresh).astype(int)
                tp = np.sum((y_val == 1) & (y_pred == 1))
                fp = np.sum((y_val == 0) & (y_pred == 1))

                tpr = tp / (total_pos + 1e-15)
                fpr = fp / (total_neg + 1e-15)

                tprs.append(tpr)
                fprs.append(fpr)

            tprs = np.array(tprs)
            fprs = np.array(fprs)
            auroc = np.trapezoid(tprs, fprs)

            plt.plot(fprs, tprs, label=f"Fold {fold} (AUROC = {auroc:.3f})")

        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve per Fold (Logistic Regression)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ROC-Curve-Log_Regr.png", dpi=300, bbox_inches='tight')
        #plt.show()



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

    for lr in [0.1, 0.05, 0.08, 0.0001]:
        for prob in [0.1, 0.2]:
            for iter in [500, 1000, 2500, 5000]:

                log_model = LogisticRegression(learning_rate=lr, max_iter=iter, prob_threshold = prob)
                y_train_binary = log_model.label_binarize(y_train)
                y_val_binary = log_model.label_binarize(y_val)

                log_model.fit(X_train_scaled, y_train_binary)

                y_val_pred = log_model.predict(X_val_scaled)
                f1_score = log_model.F1_score(y_val_binary, y_val_pred)

                y_pred_probs = log_model.predict_proba(X_val_scaled)
                auroc = log_model.get_auroc(y_val_binary, y_pred_probs)

                if f1_score > 0.90 and auroc > 0.90:
                    print(f"[LR: {lr:6.4f}, p_thresh: {prob:6.4f}, iter: {iter:4d}]",end='->')
                    print (f"f1_score = {f1_score:5.4f}", end=' -> ')
                    print (f"####auroc= {auroc:5.4f}")

        print("")

def grid_search_linregr(X_train_scaled, y_train, X_val_scaled, y_val) -> None:

    print("\n\nHyperparameter tuning for Linear Regression\n")

    print("No Regularization Grid Search Starting")
    for iter in (500, 1000, 2000):
        for lr in (0.1, 0.05, 0.001, 0.005, 0.0001):
            # No L2-Norm Regularization
            linear_model = LinearRegression(learning_rate = lr, max_iter = iter, l2_lambda=0.0)

            linear_model.fit(X_train_scaled, y_train)
            y_pred = linear_model.predict(X_val_scaled)
            rmse = linear_model.metric(y_val, y_pred)

            if (rmse < 77.0):
                print(f"[LR: {lr:5.3f}, max_iter: {iter:4d}] - RMSE = {rmse:6.4f}")

    print("No Regularization Grid Search Finished\n")

    print("L2-Norm Regularization Grid Search Starting")
    for iter in (500, 1000, 2000):
        for lr in (0.1, 0.05, 0.001, 0.005, 0.0001):
            for l2 in (0.1, 0.05, 0.01, 0.005, 0.001):
                # With L2-Norm Regularization
                linear_model = LinearRegression(learning_rate = lr, max_iter = iter, l2_lambda=l2)
                linear_model.fit(X_train_scaled, y_train)
                y_pred = linear_model.predict(X_val_scaled)
                rmse = linear_model.metric(y_val, y_pred)

                if (rmse < 77.0):
                    print(f"[LR: {lr:5.3f}, max_iter: {iter:4d} lambda: {l2:6.4f}] - RMSE = {rmse:6.4f}")

    print("L2-Norm Regularization Grid Search Finished\n")


def main():

    # =================================================================== #
    # Data handling:
    # =================================================================== #
    dp = DataProcessor("./")
    (df_train, df_test) = dp.load_data()

    # =================================================================== #
    # Data Cleaning; Drop rows with missing values
    # =================================================================== #
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
    #grid_search_linregr(X_train_scaled, y_train, X_val_scaled, y_val)

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
    #grid_search_logregr(X_train_scaled, y_train, X_val_scaled, y_val)

    log_model = LogisticRegression()
    y_train_binary = log_model.label_binarize(y_train)
    y_val_binary = log_model.label_binarize(y_val)

    log_model.fit(X_train_scaled, y_train_binary)
    y_pred = log_model.predict(X_val_scaled)

    f1_score = log_model.F1_score(y_val_binary, y_pred)

    y_pred_probs = log_model.predict_proba(X_val_scaled)
    auroc = log_model.get_auroc(y_val_binary, y_pred_probs)

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

    # Cross Validation for Linear Regression Model
    linear_model = LinearRegression()
    linear_scores = evaluator.cross_validation(linear_model, X_train, y_train)

    print("\nCross Validation")
    print("\n   Cross-Val Linear Regression: Avg RMSE:", np.mean(linear_scores))
    print("   Cross-Val Linear Regression: Std Dev RMSE:", np.std(linear_scores), "\n\n\n")

    # Cross Validation for Logistic Regression Model
    log_model = LogisticRegression()
    logistic_scores = evaluator.cross_validation(log_model, X_train, y_train_binary)

    f1_scores, aurocs = zip(*logistic_scores)
    print(f"\n   Cross-Val Log Regr: Avg F1: {np.mean(f1_scores):6.4f}, Std: {np.std(f1_scores):6.4f}")
    print(f"   Cross-Val Log Regr:  Avg AUROC: {np.mean(aurocs):6.4f}, Std: {np.std(aurocs):6.4f}")

    # =================================================================== #
    # ROC Curve
    # =================================================================== #
    evaluator = ModelEvaluator()
    evaluator.plot_roc_per_fold(log_model, X_train_scaled, y_train_binary)

    # =================================================================== #
    # Hyperparameter Tuning Linear Regression
    # =================================================================== #
    grid_search_linregr(X_train_scaled, y_train, X_val_scaled, y_val)

    # =================================================================== #
    # Hyperparameter Tuning Logistic Regression
    # =================================================================== #
    grid_search_logregr(X_train_scaled, y_train, X_val_scaled, y_val)


if __name__ == "__main__":
    main()
    print("Hello World!")

