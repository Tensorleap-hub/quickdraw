from dataset import load_dataset
# Tensorleap imports
from code_loader.default_metrics import categorical_crossentropy
from code_loader.inner_leap_binder.leapbinder_decorators import *
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.visualizers.default_visualizers import default_image_visualizer

# Preprocess Function
@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    # Load each split from the dataset (splits are handled internally)
    train_dataset = load_dataset(split='train')
    val_dataset = load_dataset(split='val')
    test_dataset = load_dataset(split='test')
    

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    train = PreprocessResponse(length=len(train_dataset), data=train_dataset)
    val = PreprocessResponse(length=len(val_dataset), data=val_dataset)
    test = PreprocessResponse(length=len(test_dataset), data=test_dataset)
    
    response = [train, val, test]
    return response

@tensorleap_input_encoder("input")
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.array(preprocess.data[idx]["image"]).astype('float32')

@tensorleap_gt_encoder("gt")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.array(preprocess.data[idx]["label"]).astype('float32')

@tensorleap_metadata("gt_label_name")
def metadata_label(idx: int, preprocess: PreprocessResponse) -> str:
    label_name = preprocess.data[idx]["label_name"]
    return label_name

@tensorleap_metadata("gt_label_id")
def metadata_label_id(idx: int, preprocess: PreprocessResponse) -> str:
    label_id = preprocess.data[idx]["label"]
    return label_id

@tensorleap_metadata("country")
def metadata_country(idx: int, preprocess: PreprocessResponse) -> str:
    country = preprocess.data[idx]['country']
    country = str(country) if country else 'UNK'  # Handle missing values
    return country

@tensorleap_metadata("recognized")
def metadata_recognized(idx: int, preprocess: PreprocessResponse) -> bool:
    recognized = preprocess.data[idx]['recognized']
    return bool(recognized)

@tensorleap_metadata("timestamp")
def metadata_timestamp(idx: int, preprocess: PreprocessResponse) -> float:
    return float(preprocess.data[idx]['timestamp'])

@tensorleap_metadata("word")
def metadata_word(idx: int, preprocess: PreprocessResponse) -> str:
    return str(preprocess.data[idx].get('word', ''))

@tensorleap_metadata("num_strokes")
def metadata_num_strokes(idx: int, preprocess: PreprocessResponse) -> int:
    return int(preprocess.data[idx]['num_strokes'])

@tensorleap_custom_metric("predicted_label_id", compute_insights=False)
def predicted_label_name(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
    y_pred_label_int = np.float32(np.argmax(y_pred, axis=-1))
    return y_pred_label_int

@tensorleap_custom_metric("sample_accuracy", direction=MetricDirection.Upward)
def custom_metric_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
    y_pred_labels = np.argmax(y_pred, axis=-1)
    y_true_fixed = y_true.reshape(-1)
    y_pred_labels = y_pred_labels.reshape(-1)
    correct_predictions = np.array(y_pred_labels == y_true_fixed).astype(np.float32)
    return correct_predictions

@tensorleap_custom_loss("categorical_crossentropy")
def custom_loss_categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
    # softmax on y_pred
    y_pred_softmaxed = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
    #convert y_true to one-hot encoding
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
    return categorical_crossentropy(y_true_one_hot, y_pred_softmaxed)

@tensorleap_custom_visualizer('default_image_visualizer', LeapDataType.Image)
def image_visualizer(data: np.float32):
    return default_image_visualizer(data.reshape(1,28,28,1))
