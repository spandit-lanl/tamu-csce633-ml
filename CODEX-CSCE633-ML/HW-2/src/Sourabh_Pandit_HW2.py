import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

skip_grid_search = True  # Global flag, set to true when running grid_search
hyp_idx = -1            # Index into hyperparam list
hyp_list = []           # List for hyperparameter tuning in grid_search

# Grid Search Options
for ru in [False, True]:                      # replace unknown values ?
    # , True]:           # drop cols (day, duration, and default?
    for drop in [False]:
        for upsample in [False]:  # , True]:    # Upsample the data for minority class ?
            for depth in range(4, 12):                 # max tree depth
                for split_size in range(5, 20, 2):       # mininum sample split size
                    # use_entropy (False meanis use gini index)
                    for entropy in [False, True]:
                        hyp_list.append(
                            (ru, drop, upsample, depth, split_size, entropy))


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


'''
Problem A-1: Data Preprocessing and EDA
'''


class DataLoader:
    # Static variables
    st_categorical_cols = []
    st_categorical_indices = []
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class
    will be tested on the pre-built enviornment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int):
        '''
        Inialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.data = pd.read_csv(data_root, delimiter=';')
        self.data_train = None
        self.data_valid = None

        # TODO: Hyperparam and flag
        if skip_grid_search == True:
            self.replace_unknown = False
            self.drop_cols = False
            self.upsample_train_data = False  # True
        else:
            self.replace_unknown = hyp_list[hyp_idx][0]
            self.drop_cols = hyp_list[hyp_idx][1]
            self.upsample_train_data = hyp_list[hyp_idx][2]

        # TODO: Revisit this ...
        self.prep_done = False
        self.data_prep()
        self.data_split()

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20.
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''

        if self.data_train is not None and self.data_valid is not None:
            return

        y_unique_values = set(self.data['y'].unique())

        if y_unique_values == {'yes', 'no'}:
            pos_val = 'yes'
            neg_val = 'no'
        elif y_unique_values == {0, 1}:
            pos_val = 1
            neg_val = 0

        # Train and Validation Split using Strata: Separate class neg_val and class pos_val
        class_neg = self.data[self.data['y'] == neg_val]
        class_pos = self.data[self.data['y'] == pos_val]

        # Shuffle each class, all samples
        class_neg = class_neg.sample(
            frac=1, random_state=self.random_state).reset_index(drop=True)
        class_pos = class_pos.sample(
            frac=1, random_state=self.random_state).reset_index(drop=True)

        # Indices to split each class in 80/20
        split_idx_neg = int(0.8 * len(class_neg))
        split_idx_pos = int(0.8 * len(class_pos))

        # Training and Validation data from class neg_val
        train_neg = class_neg.iloc[:split_idx_neg]
        valid_neg = class_neg.iloc[split_idx_neg:]

        # Training and Validation data from class pos_val
        train_pos = class_pos.iloc[:split_idx_pos]
        valid_pos = class_pos.iloc[split_idx_pos:]

        # Combine the training data from two classes and shuffle again
        self.data_train = pd.concat(
            [train_neg, train_pos]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Combine the validation data from two classes and shuffle again
        self.data_valid = pd.concat(
            [valid_neg, valid_pos]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # TODO: Upsampling/Oversampling minority class
        # We have imbalanced data and need to upsample the 'yes' y-values
        if self.upsample_train_data == True:
            pos = self.data_train[self.data_train['y'] == pos_val]
            neg = self.data_train[self.data_train['y'] == neg_val]

            # Oversample positives to match negatives
            pos_upsampled = pos.sample(
                n=int(len(neg)), replace=True, random_state=self.random_state)

            # Recombine and shuffle
            self.data_train = pd.concat(
                [neg, pos_upsampled]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values.
        '''

        if self.prep_done == True:
            return

        df = self.data.copy()                   # Work on a safe copy
        df.columns = df.columns.str.strip()     # Clean column names

        # TODO; check if duration and default should be dropped or kept
        for col in ['day', 'duration', 'default']:
            if self.drop_cols and col in df.columns:
                df.drop(columns=[col], inplace=True)

        if self.replace_unknown:
            for col in df.select_dtypes(include=['object']).columns:
                unknown_count = (df[col] == 'unknown').sum()
                unknown_ratio = unknown_count / len(df)
                if unknown_count > 0 and unknown_ratio < 0.05:
                    mode = df[df[col] != 'unknown'][col].mode()
                    if not mode.empty:
                        df.loc[:, col] = df[col].replace('unknown', mode[0])

        df.dropna(inplace=True)                 # Drop any remaining NaNs
        df.reset_index(drop=True, inplace=True)

        DataLoader.st_categorical_cols = df.select_dtypes(
            include=['object']).columns
        DataLoader.st_categorical_indices = [df.columns.get_loc(
            col) for col in DataLoader.st_categorical_cols]

        # Map categorical data to numerical values
        for col in df.select_dtypes(include=['object']).columns:
            df[col], _ = pd.factorize(df[col])

        self.data = df

    """
    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values.
        '''

        if self.prep_done == True:
            return

        def _data_prep(df, set_cat_col_indices:bool=False):
            df = df.copy()
            df.columns = df.columns.str.strip()

            # Drop specified columns if in drop list
            for col in ['day', 'duration', 'default']:
                if self.drop_cols and col in df.columns:
                    df.drop(columns=[col], inplace=True)

            # Replace 'unknown' with mode if rare
            if self.replace_unknown:
                for col in df.select_dtypes(include=['object']).columns:
                    unknown_count = (df[col] == 'unknown').sum()
                    unknown_ratio = unknown_count / len(df)
                    if unknown_count > 0 and unknown_ratio < 0.05:
                        mode = df[df[col] != 'unknown'][col].mode()
                        if not mode.empty:
                            df.loc[:, col] = df[col].replace('unknown', mode[0])

            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Save categorical column info
            if set_cat_col_indices:
                DataLoader.st_categorical_cols = df.select_dtypes(include=['object']).columns
                DataLoader.st_categorical_indices = [df.columns.get_loc(col) for col in DataLoader.st_categorical_cols]

            # Convert categorical columns to numeric
            for col in DataLoader.st_categorical_cols:
                df[col], _ = pd.factorize(df[col])

            return df


        #if self.data_train is None  or self.data_valid is None:
        #    self.data_split()

        # Apply cleaning to all datasets
        self.data = _data_prep(self.data, set_cat_col_indices=True)
        self.data_train = _data_prep(self.data_train)
        self.data_valid = _data_prep(self.data_valid)

        self.prep_done = True
    """

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test
        data.

        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''

        X = data.drop(columns=['y']).values
        y = data['y'].values
        return (X, y)

    def plot_histogram(self):

        cols = self.data.columns
        num_cols = len(cols)
        plot_per_row = 4  # Number of plots per row

        if (num_cols/plot_per_row) > int(num_cols/plot_per_row):
            num_plot_rows = int(num_cols / plot_per_row + 1)
        else:
            num_plot_rows = int(num_cols / plot_per_row)

        fig, axes = plt.subplots(num_plot_rows, plot_per_row, figsize=(
            5 * plot_per_row, 4 * num_plot_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            if self.data[col].dtype == 'object' or self.data[col].nunique() < 10:
                # Bar plot for categorical
                self.data[col].value_counts().plot(
                    kind='bar', ax=ax, edgecolor='black')
                ax.set_ylabel("Count")
            else:
                # Histogram for numeric
                self.data[col].plot(kind='hist', bins=30,
                                    ax=ax, edgecolor='black')
                ax.set_ylabel("Frequency")

            ax.set_title(col)
            ax.set_xlabel(col)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig("histogram.png", dpi=300, format="png")
        # plt.show()


'''
Porblem A-2: Classification Tree Inplementation
'''


class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built enviornment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''

        def __init__(self, split=None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value, is_categorical)
                - For numerical features: split_value is the threshold
                - For categorical features: split_value is a set of categories for the left branch
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            '''

            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction

            if split is not None:
                self.feature_idx = split[0]
                self.split_value = split[1]
                self.is_categorical = split[2]

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int, max_depth: int = 0):

        self.random_state = random_state
        np.random.seed(self.random_state)
        self.tree_root = None
        self.categorical_indices = DataLoader.st_categorical_indices

        # TODO: Hyperparam and flag
        if skip_grid_search == True:
            self.max_depth = 7
            self.min_samples_split = 3
            self.use_entropy = False  # True
        else:
            self.max_depth = hyp_list[hyp_idx][3]
            self.min_samples_split = hyp_list[hyp_idx][4]
            self.use_entropy = hyp_list[hyp_idx][5]

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Computes impurity of labels y using the specified method.

        Args:
            y (np.ndarray): array of labels
            method (str): "gini" or "entropy" (default: "entropy")

        Returns:
            float: impurity score
        '''

        if self.use_entropy == True:
            return self.entropy(y)
        else:
            return self.gini_index(y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.build_tree(X, y)

    def build_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Public method to initiate tree construction. Stores the root node.
        '''

        def _build_tree_recursive(X: np.ndarray, y: np.ndarray, depth: int):
            # At a pure node (all labels are the same), return a leaf node
            if np.unique(y).size == 1:
                return self.Node(prediction=y[0])

            # Stop condition: not enough samples or max depth reached
            if len(y) < self.min_samples_split or depth >= self.max_depth:
                majority_class = np.bincount(y.astype(int)).argmax()
                return self.Node(prediction=majority_class)

            # Find the best split
            split_result = self.search_best_split(X, y)

            # If no valid split, return a leaf with majority class
            if split_result is None:
                majority_class = np.bincount(y.astype(int)).argmax()
                return self.Node(prediction=majority_class)

            feature_index, split_value, is_categorical = split_result
            X_left, y_left, X_right, y_right = self.split(
                X, y, feature_index, split_value, is_categorical)

            # If split failed to divide the data, return leaf
            if X_left is None or X_right is None:
                majority_class = np.bincount(y.astype(int)).argmax()
                return self.Node(prediction=majority_class)

            # Recursively build the left and right subtrees
            left_child = _build_tree_recursive(X_left, y_left, depth + 1)
            right_child = _build_tree_recursive(X_right, y_right, depth + 1)

            # Return the internal node
            return self.Node(
                split=(feature_index, split_value, is_categorical),
                left=left_child,
                right=right_child,
                prediction=None
            )

        # Start tree construction
        self.tree_root = _build_tree_recursive(X, y, depth=0)
        return self.tree_root

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Search for the best feature and split value.

        Returns:
        (feature_index, split_value, is_categorical) if a split is found,
        else None
        '''

        best_gain = -1
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            is_categorical = feature_index in self.categorical_indices

            if is_categorical:
                # ---------- SIMPLE CATEGORICAL SPLITTING ----------
                for val in unique_values:
                    left_set = {val}
                    X_left, y_left, X_right, y_right = self.split(
                        X, y, feature_index, left_set, is_categorical=True
                    )
                    if X_left is None or X_right is None:
                        continue
                    gain = self.information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, left_set, True)

                '''
                # ---------- EXHAUSTIVE CATEGORICAL SPLITTING ----------
                num_vals = len(unique_values)
                for mask in range(1, 2 ** num_vals - 1):
                    left_set = {unique_values[i] for i in range(num_vals) if (mask >> i) & 1}
                    X_left, y_left, X_right, y_right = self.split(
                        X, y, feature_index, left_set, is_categorical=True
                    )
                    if X_left is None or X_right is None:
                        continue
                    gain = self.information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, left_set, True)
                '''
            else:
                sorted_vals = np.sort(unique_values)
                for i in range(1, len(sorted_vals)):
                    threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2
                    X_left, y_left, X_right, y_right = self.split(
                        X, y, feature_index, threshold, is_categorical=False
                    )
                    if X_left is None or X_right is None:
                        continue
                    gain = self.information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, threshold, False)

        return best_split

    def predict(self, X: np.ndarray, dbg_print: bool = True) -> np.ndarray:
        '''
        Predict classes for multiple samples.

        Args:
            X: numpy array with the same columns as the training data

        Returns:
            np.ndarray: Array of predictions
        '''

        if dbg_print:
            print(f"DEBUG: Enter predict()", flush=True)

        def _predict_one(x: np.ndarray, node) -> int:
            # Recursively predict the class for a single sample.
            if node.is_leaf():
                return node.prediction

            if node.is_categorical:
                if x[node.feature_idx] in node.split_value:
                    return _predict_one(x, node.left)
                else:
                    return _predict_one(x, node.right)
            else:
                if x[node.feature_idx] <= node.split_value:
                    return _predict_one(x, node.left)
                else:
                    return _predict_one(x, node.right)

        ret = np.array([_predict_one(x, self.tree_root) for x in X])
        return ret

    def entropy(self, y):
        entropy = 0
        class_labels = np.unique(y)

        for cls in class_labels:
            p_cls = len(y[y == cls])/len(y)
            entropy -= p_cls * np.log2(p_cls)

        return entropy

    def gini_index(self, y):
        gini = 0
        class_labels = np.unique(y)

        for cls in class_labels:
            p_cls = len(y[y == cls])/len(y)
            gini += p_cls**2

        return (1 - gini)

    def split(self, X: np.ndarray, y: np.ndarray, feature_index: int, split_value, is_categorical: bool):
        '''
        Splits the dataset (X, y) based on the feature at feature_index.

        If is_categorical is True:
            - split_value is a set of category values for the left branch
        If is_categorical is False:
            - split_value is a numeric threshold

        Returns:
            X_left, y_left, X_right, y_right
        '''

        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        if is_categorical:
            left_rows = [
                row for row in dataset if row[feature_index] in split_value]
            right_rows = [
                row for row in dataset if row[feature_index] not in split_value]
        else:
            left_rows = [
                row for row in dataset if row[feature_index] <= split_value]
            right_rows = [
                row for row in dataset if row[feature_index] > split_value]

        dataset_left = np.array(left_rows)
        dataset_right = np.array(right_rows)

        if len(dataset_left) == 0 or len(dataset_right) == 0:
            return None, None, None, None

        X_left, y_left = dataset_left[:, :-1], dataset_left[:, -1]
        X_right, y_right = dataset_right[:, :-1], dataset_right[:, -1]

        return X_left, y_left, X_right, y_right

    def information_gain(self, y, y_left, y_right):
        '''
        Computes information gain from a proposed split.

        Args:
            y: Full label array before the split
            y_left, y_right: Label arrays after split
            method: "gini" or "entropy"

        Returns:
            Information gain (float)
        '''
        parent_impurity = self.split_crit(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0

        child_impurity = (
            n_left * self.split_crit(y_left) +
            n_right * self.split_crit(y_right)
        ) / n

        return parent_impurity - child_impurity

    def print_tree(self, node=None, depth=0):
        """
        Recursively print the structure of the decision tree.
        Args:
            node: Node to print (defaults to tree root)
            depth: Current depth in the tree (used for indentation)
        """
        if node is None:
            node = self.tree_root

        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}Leaf: Predict {node.prediction}")
        else:
            feature = node.feature_idx
            if node.is_categorical:
                print(f"{indent}Feature[{feature}] in {node.split_value}?")
            else:
                print(f"{indent}Feature[{feature}] <= {node.split_value}?")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    @staticmethod
    def precision(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        Compute the F1-score for binary classification (0/1 labels).

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            F1-score as a float
        '''
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def grid_search():

        global hyp_idx
        global skip_grid_search

        skip_grid_search = False

        print("\tMyGridSearch: Replace_uknown,drop_cols,upsample_data,max_depth,min_split_size,use_entropy,",
              end='')

        print("Accuracy,Precision,Recall,F1-SCore")
        for hyp_idx in range(0, len(hyp_list)):
            loader = DataLoader("./bank.csv", 42)
            # loader.plot_histogram()

            loader.data_split()
            loader.data_prep()

            X_train, y_train = loader.extract_features_and_label(
                loader.data_train)
            X_valid, y_valid = loader.extract_features_and_label(
                loader.data_valid)

            decision_tree = ClassificationTree(random_state=42)
            decision_tree.build_tree(X_train, y_train)
            y_pred = decision_tree.predict(X_valid, dbg_print=False)

            accuracy = (y_pred == y_valid).mean()
            prec = ClassificationTree.precision(y_valid, y_pred)
            rec = ClassificationTree.recall(y_valid, y_pred)
            f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)

            print(f"\tMyGridSearch: Run {hyp_idx}/{len(hyp_list)} {hyp_list[hyp_idx]},{accuracy:.4f},{prec:.4f},{rec:.4f},{f1:.4f}",
                  flush=True)

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path="roc_curve.png"):
        '''
        Compute and plot ROC curve using TPR/FPR at various thresholds.
        '''

        print(f"\nGetting Ready to plot ROC AUC curve")
        # Sort by predicted probabilities descending
        desc_sort = np.argsort(-y_prob)
        y_true = y_true[desc_sort]
        y_prob = y_prob[desc_sort]

        # Total positives and negatives
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)

        tpr_list = []
        fpr_list = []

        tp = fp = 0
        for i in range(len(y_true)):
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / P if P > 0 else 0
            fpr = fp / N if N > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr_list, fpr_list)

        # Plot ROC
        plt.figure(figsize=(6, 4))
        plt.plot(fpr_list, tpr_list,
                 label=f"ROC Curve (AUC = {auc:.4f})", color="blue")
        plt.plot([0, 1], [0, 1], linestyle='--',
                 color='gray', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for my_best_model")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, format="png")

        print(f"\n\tROC AUC (Area Under the Curver: {auc:.4f}")


def train_XGBoost(dbg_print: bool = True) -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''

    if dbg_print:
        print(f"DEBUG: train_XGBoost(): Enter")

    alpha_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    results = {}

    loader = DataLoader("./bank.csv", random_state=42)
    loader.data_split()
    loader.data_prep()

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    for alpha in alpha_vals:
        f1_scores = []

        for i in range(100):
            # Sample with replacement
            boot_idx = np.random.choice(
                len(X_train), size=len(X_train), replace=True)
            X_bootstrap = X_train[boot_idx]
            y_bootstrap = y_train[boot_idx]

            model = XGBClassifier(
                max_depth=8,
                n_estimators=100,
                eval_metric='logloss',
                reg_lambda=alpha,
                random_state=42,
                n_jobs=1
            )

            model.fit(X_bootstrap, y_bootstrap)
            y_pred = model.predict(X_valid)
            f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)
            f1_scores.append(f1)

        avg_f1 = np.mean(f1_scores)
        results[alpha] = avg_f1
        print(f"\tAlpha = {alpha:8.3f}: Avg F1 = {avg_f1:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xscale('log')
    plt.xlabel("Alpha (L2 Regularization Strength)")
    plt.ylabel("Average F1 Score")
    plt.title("Avg F1 Score vs. Alpha (reg_lambda)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("xgboost_best_alpha.png", dpi=300, format='png')
    # plt.show()

    best_alpha = max(results, key=results.get)
    print(
        f"\n\tBest alpha: {best_alpha} with Avg F1 = {results[best_alpha]:.4f}\n")

    return {
        "alpha_scores": results,
        "best_alpha": best_alpha
    }


'''
Initialize the following variable with the best model you have found. This model will be used in testing
in our pre-built environment.
'''
my_best_model = XGBClassifier(
    n_estimators=100,
    eval_metric='logloss',
    reg_lambda=0.01,
    random_state=42,
    n_jobs=1)


def test_decision_tree():

    loader = DataLoader("./bank.csv", 42)
    loader.data_split()
    loader.data_prep()
    print("\nFirst 10 rows/samples of the training data")
    print(loader.data_train.head(11))

    print("\nNow actually running train and predict with my Decison Tree")

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    decision_tree = ClassificationTree(random_state=42)
    decision_tree.build_tree(X_train, y_train)

    y_pred = decision_tree.predict(X_valid, dbg_print=False)
    accuracy = (y_pred == y_valid).mean()
    prec = ClassificationTree.precision(y_valid, y_pred)
    rec = ClassificationTree.recall(y_valid, y_pred)
    f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)

    print(f"\tAccuracy  : {accuracy:.4f}")
    print(f"\tPrecision : {prec:.4f}")
    print(f"\tRecall    : {rec:.4f}")
    print(f"\tF1 Score  : {f1:.4f}", flush=True)


def run_grid_search():
    ClassificationTree.grid_search()


def test_train_xgboost():

    print(f"\n\tSearch for best alpha with XG boost")
    results = train_XGBoost(dbg_print=False)
    best_alpha = results["best_alpha"]

    loader = DataLoader("./bank.csv", random_state=42)
    loader.data_split()
    loader.data_prep()

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    # Initialize XGBClassifier with the best_alpha returned from train_XGBoost()
    print(f"\nNow Running the best XG Boost model with alpha = {best_alpha}")
    my_best_model = XGBClassifier(
        n_estimators=100,
        eval_metric='logloss',
        reg_lambda=best_alpha,
        random_state=42,
        n_jobs=1)

    my_best_model.fit(X_train, y_train)

    # Predict
    y_pred = my_best_model.predict(X_valid)

    # Evaluate
    accuracy = (y_pred == y_valid).mean()
    prec = ClassificationTree.precision(y_valid, y_pred)
    rec = ClassificationTree.recall(y_valid, y_pred)
    f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)

    print(f"\n\tmy_best_model Evaluation:")
    print(f"\t\tAccuracy  : {accuracy:.4f}")
    print(f"\t\tPrecision : {prec:.4f}")
    print(f"\t\tRecall    : {rec:.4f}")
    print(f"\t\tF1 Score  : {f1:.4f}")

    # Predict probabilities for positive class
    y_prob = my_best_model.predict_proba(X_valid)[:, 1]
    ClassificationTree.plot_roc_curve(y_valid, y_prob, "roc_auc.png")


def test_my_best_model():

    print(f"\n\tTest my_best_model")

    loader = DataLoader("./bank.csv", random_state=42)
    loader.data_split()
    loader.data_prep()

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    global my_best_model
    my_best_model.fit(X_train, y_train)

    # Predict
    y_pred = my_best_model.predict(X_valid)

    # Evaluate
    accuracy = (y_pred == y_valid).mean()
    prec = ClassificationTree.precision(y_valid, y_pred)
    rec = ClassificationTree.recall(y_valid, y_pred)
    f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)

    print(f"\n\tmy_best_model Evaluation:")
    print(f"\t\tAccuracy  : {accuracy:.4f}")
    print(f"\t\tPrecision : {prec:.4f}")
    print(f"\t\tRecall    : {rec:.4f}")
    print(f"\t\tF1 Score  : {f1:.4f}")

    # Predict probabilities for positive class
    y_prob = my_best_model.predict_proba(X_valid)[:, 1]
    ClassificationTree.plot_roc_curve(y_valid, y_prob, "roc_auc.png")


def main():
    print("\nFirst running train and predict with my Decison Tree")
    test_decision_tree()

    print("\nSecond: Test XG boost")
    test_train_xgboost()

    print("\nThird: Test global my_best_model")
    test_my_best_model()

    print("\nFinally: GridSearch for hyperparame tuning for -Accuracy, Precision, Recall, F1-Score")
    run_grid_search()


if __name__ == "__main__":
    main()
