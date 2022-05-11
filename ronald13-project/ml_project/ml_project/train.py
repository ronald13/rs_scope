from pathlib import Path
from joblib import dump
import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from .data import get_dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--search",
    default="NestedCV",
    type=click.Choice(["KFold", "NestedCV"]),
    show_default=True,
)
@click.option(
    "--select-model",
    default="logist_regression",
    type=click.Choice(["logist_regression", "random_forest"]),
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)

@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=10,
    type=float,
    show_default=True,
)

@click.option(
    "--feature-engineering",
    default="standard_scaler",
    type=click.Choice(["standard_scaler", "min_max_scaler", "None"]),
    show_default=True,
)

@click.option(
    "--n-estimators",
    default=20,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--max-features",
    default=1,
    type=click.FloatRange(0, 1, min_open=True, max_open=False),
    show_default=True,
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    feature_engineering: str,
    max_iter: int,
    logreg_c: float,
    select_model: str,
    search: str,
    n_estimators: int,
    max_depth: int,
    max_features: float,

) -> None:
    features_train, target_train = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio
    )
    with mlflow.start_run():
        if feature_engineering == "standard_scaler":
            features_train = StandardScaler().fit_transform(features_train)
        elif feature_engineering == "min_max_scaler":
            features_train = MinMaxScaler().fit_transform(features_train)
        cv = KFold(n_splits=8, shuffle=True, random_state=random_state)

        if search == "KFold":

            if select_model == "logist_regression":
                model = LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_c, n_jobs=-1
                )
            if select_model == "random_forest":
                model = RandomForestClassifier(
                    random_state=random_state,
                    max_depth=max_depth,
                    max_features=max_features,
                    n_estimators=n_estimators,
                    n_jobs=-1,
                )

            scores = cross_validate(
                model,
                features_train,
                target_train,
                cv=cv,
                scoring=("accuracy", "roc_auc_ovr", "f1_macro"),
            )
            accuracy = scores["test_accuracy"].mean()
            roc_auc_ovr = scores["test_roc_auc_ovr"].mean()
            f1_macro = scores["test_f1_macro"].mean()


            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)

        elif search == "NestedCV":
            model = LogisticRegression(random_state=random_state, n_jobs=-1)
            space = dict()
            space["max_iter"] = [10, 100, 200, 400, 500, 700]
            space["C"] = [0.001, 0.01, 0.1, 1, 10]

            search_q = GridSearchCV(model, space, scoring="accuracy", n_jobs=-1, cv=cv)
            cv_outer = KFold(n_splits=5)
            scores = cross_validate(
                search_q,
                features_train,
                target_train,
                cv=cv_outer,
                scoring=("accuracy", "roc_auc_ovr", "f1_macro"),
            )
            accuracy = scores["test_accuracy"].mean()
            roc_auc_ovr = scores["test_roc_auc_ovr"].mean()
            f1_macro = scores["test_f1_macro"].mean()

            search_q.fit(features_train, target_train)
            print(
                "The parameters combination that would give best accuracy is : ",
                search_q.best_params_,
            )
            mlflow.log_param("max_iter", search_q.best_params_["max_iter"])
            mlflow.log_param("logreg_c", search_q.best_params_["C"])

        mlflow.log_param("feature_engineering", feature_engineering)
        mlflow.log_param("select_model", select_model)
        mlflow.log_param("search", search)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc_ovr", roc_auc_ovr)
        mlflow.log_metric("f1_macro", f1_macro)
        click.echo(f"accuracy: {accuracy}.")
        click.echo(f"roc_auc_ovr: {roc_auc_ovr}.")
        click.echo(f"f1_macro: {f1_macro}.")
        click.echo(f"select_model: {select_model}.")
        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")