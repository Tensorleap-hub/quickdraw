from dataset import load_dataset
# Tensorleap imports
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

@tensorleap_metadata("label")
def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int

@tensorleap_metadata("country")
def metadata_country(idx: int, preprocess: PreprocessResponse) -> str:
    country = preprocess.data[idx]['country']
    country = str(country) if country else 'UNK'  # Handle missing values
    return country

@tensorleap_metadata("recognized")
def metadata_recognized(idx: int, preprocess: PreprocessResponse) -> bool:
    recognized = preprocess.data[idx]['recognized']
    return bool(recognized)

@tensorleap_custom_metric("accuracy")
def custom_metric_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:

    y_pred_labels = np.argmax(y_pred, axis=-1)
    y_true_labels = np.argmax(y_true, axis=-1)
    correct_predictions = np.array(y_pred_labels == y_true_labels).astype(np.float32)
    return correct_predictions

@tensorleap_custom_loss("categorical_crossentropy")
def custom_loss_categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
    # categorical crossentropy loss implementation - returns one value per sample
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
    return loss.astype(np.float32)

@tensorleap_custom_visualizer('default_image_visualizer', LeapDataType.Image)
def image_visualizer(data: np.float32):
    return default_image_visualizer(data.reshape(1,28,28,1))
