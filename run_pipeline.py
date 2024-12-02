import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_fn(fn_args):
    """Train an LSTM model for the Erroneous Dataset."""
    train_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=fn_args.train_files,
        batch_size=32,
        features_spec=fn_args.schema.as_feature_spec(),
        reader=tf.data.TFRecordDataset
    )
    eval_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=fn_args.eval_files,
        batch_size=32,
        features_spec=fn_args.schema.as_feature_spec(),
        reader=tf.data.TFRecordDataset
    )

    model = Sequential([
        LSTM(32, input_shape=(1, 2), return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(train_dataset, validation_data=eval_dataset, epochs=10)
    model.save(fn_args.serving_model_dir)

from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from pipelines.mock_pipeline import create_pipeline as create_mock_pipeline
from pipelines.erroneous_pipeline import create_pipeline as create_erroneous_pipeline

# Paths
MOCK_PIPELINE_ROOT = 'mock_pipeline_root'
ERRONEOUS_PIPELINE_ROOT = 'erroneous_pipeline_root'
MOCK_DATA_ROOT = 'mock_data_root'
ERRONEOUS_DATA_ROOT = 'erroneous_data_root'
MOCK_MODULE_FILE = 'mock_pipeline/trainer.py'
ERRONEOUS_MODULE_FILE = 'erroneous_pipeline/trainer.py'
MOCK_SERVING_MODEL_DIR = 'mock_serving_model'
ERRONEOUS_SERVING_MODEL_DIR = 'erroneous_serving_model'

# Run Mock Pipeline
mock_pipeline = create_mock_pipeline(
    pipeline_name='mock_pipeline',
    pipeline_root=MOCK_PIPELINE_ROOT,
    data_root=MOCK_DATA_ROOT,
    module_file=MOCK_MODULE_FILE,
    serving_model_dir=MOCK_SERVING_MODEL_DIR
)
LocalDagRunner().run(mock_pipeline)

# Run Erroneous Pipeline
erroneous_pipeline = create_erroneous_pipeline(
    pipeline_name='erroneous_pipeline',
    pipeline_root=ERRONEOUS_PIPELINE_ROOT,
    data_root=ERRONEOUS_DATA_ROOT,
    module_file=ERRONEOUS_MODULE_FILE,
    serving_model_dir=ERRONEOUS_SERVING_MODEL_DIR
)
LocalDagRunner().run(erroneous_pipeline)
