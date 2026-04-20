# %%
#Install Required Dependencies
import subprocess
subprocess.run([
    "pip", "install",
    "keras~=3.8.0",
    "matplotlib~=3.10.0",
    "numpy~=2.0.0",
    "pandas~=2.2.0",
    "tensorflow~=2.18.0",
    "kaleido==0.2.1",
    "seaborn~=0.13.0",
])

# %%
#Load dependencies

#Data
import numpy as np
import pandas as pd

#Machine Learning
import keras

#Data Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


from dataclasses import dataclass, field

# %%
#Hyperparameters for the model
@dataclass
class ModelSettings:
    learning_rate: float
    number_epochs: int
    batch_size: int
    input_features: list[str] = field(default_factory=list)

# %%
#Load & Prepare the Data
chicago_taxi_dataset = pd.read_csv(
  "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"  
)

training_df = chicago_taxi_dataset[
    ["TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE"]
].copy()

print(f"Rows loaded: {len(training_df)}")
print(training_df.describe(include='all'))

# %%
#Exploratory Plots
def plot_scatter_matrix(df: pd.DataFrame, columns: list[str]) -> None:
    sns.pairplot(df[columns])
    plt.suptitle("Pairwise Relationships", y=1.02)
    plt.savefig("PairwiseRelationships.png", bbox_inches="tight")
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("CorrelationMatrix.png")
    plt.show()

plot_scatter_matrix(training_df, ["FARE", "TRIP_MILES", "TRIP_SECONDS"])
plot_correlation_matrix(training_df)

# %%
#Model Creation
from tensorflow import shape


def create_model(
    settings: ModelSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """
    Builds and complies a single-layer linear model.
    One input node per feature -> Concatenate -> Dense(1) output
    """
    inputs = {
        name: keras.Input(shape=(1,), name=name)
        for name in settings.input_features
    }

    if len(inputs) == 1:
        # Concatenate requires ≥2 tensors; skip it for single-feature models.
        x = list(inputs.values())[0]
    else:
        x = keras.layers.Concatenate()(list(inputs.values()))    
  
    
    outputs = keras.layers.Dense(units=1, name="prediction")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=metrics,
    )
    return model

# %%
#Training the Model
def train_model(
        model_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        label_name: str,
        settings: ModelSettings,
) -> dict:
    """
    Trains the model and returns a plain dictionary with everything needed for evaluation and plotting
    """
    features = {
        name: dataset[name].to_numpy()
        for name in settings.input_features
    }
    labels = dataset[label_name].to_numpy()

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
        verbose=1,
    )

    return{
        "name": model_name,
        "model": model,
        "settings": settings,
        "history": pd.DataFrame(history.history),
        "epochs": history.epoch,
    }

# %%
#Results Plot
def plot_training_metrics(training: dict, metric_names: list[str]) -> None:
    """Line chart for one or more training metrics over epochs."""
    history = training["history"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metric_names:
        if metric in history.columns:
            ax.plot(training["epochs"], history[metric], marker="o", label=metric)

    ax.set_title(f"Training Metrics — {training['name']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("TrainingMetrics.png")
    plt.show()


def plot_predictions(
    training: dict,
    dataset: pd.DataFrame,
    label_name: str,
    sample_size: int = 200,
) -> None:
    sample = dataset.sample(n=min(sample_size, len(dataset)), random_state=42).copy()
    features = {
        name: sample[name].to_numpy()
        for name in training["settings"].input_features
    }
    predictions = training["model"].predict(features, verbose=0).flatten()
    sample["_predicted"] = predictions

    if len(training["settings"].input_features) == 2:
        x_feat, y_feat = training["settings"].input_features

        # Build grid and predict surface
        x_range = np.linspace(sample[x_feat].min(), sample[x_feat].max(), 50)
        y_range = np.linspace(sample[y_feat].min(), sample[y_feat].max(), 50)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = training["model"].predict(
            {x_feat: xx.ravel(), y_feat: yy.ravel()}, verbose=0
        ).reshape(xx.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Prediction surface
        ax.plot_surface(xx, yy, zz, cmap="plasma", alpha=0.6, edgecolor="none")

        # Actual data points
        ax.scatter(
            sample[x_feat],
            sample[y_feat],
            sample[label_name],
            color="steelblue",
            s=20,
            alpha=0.6,
            label="Actual",
        )

        ax.set_title(f"Prediction Surface vs Actual — {training['name']}")
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_zlabel(label_name)
        ax.legend()

    else:
        # 2D fallback for single feature
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sample[label_name], predictions, alpha=0.5, label="Predictions")

        min_val = min(sample[label_name].min(), predictions.min())
        max_val = max(sample[label_name].max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color="red", linestyle="--", label="Perfect fit")

        ax.set_title(f"Predicted vs Actual — {training['name']}")
        ax.set_xlabel(f"Actual {label_name}")
        ax.set_ylabel(f"Predicted {label_name}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("PredictionPlot.png")
    plt.show()

# %%
#Run Training and Plot Results
settings_1 = ModelSettings(
    learning_rate=0.001,
    number_epochs=20,
    batch_size=50,
    input_features=["TRIP_MILES"],
)

metrics = [keras.metrics.RootMeanSquaredError(name="rmse")]

model_1 = create_model(settings_1, metrics)
model_1.summary()

training_1 = train_model(
    model_name="one_feature",
    model=model_1,
    dataset=training_df,
    label_name="FARE",
    settings=settings_1,
)

plot_training_metrics(training_1, ["rmse"])
plot_predictions(training_1, training_df, "FARE")

# %%
#Two Features
#Add derived feature for trip duration in minutes, since the original dataset only has trip duration in seconds, and minutes may be more interpretable and have a more linear relationship with fare.
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS'] / 60

#Hyperparameters for the model
settings_2 = ModelSettings(
    learning_rate=0.001,
    number_epochs=20,
    batch_size=50,
    input_features=["TRIP_MILES", "TRIP_MINUTES"],
)

metrics = [keras.metrics.RootMeanSquaredError(name="rmse")] 

model_2 = create_model(settings_2, metrics)

training_2 = train_model("two_features", model_2, training_df, "FARE", settings_2)

plot_training_metrics(training_2, ["rmse"])
plot_predictions(training_2, training_df, "FARE")

# %%



