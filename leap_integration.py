from code_loader.plot_functions.visualize import visualize
from leap_binder import *
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
import onnxruntime as ort
from data_download import classes

prediction_type1 = PredictionTypeHandler('classes', classes, channel_dim=-1)

@tensorleap_load_model([prediction_type1])
def load_model():
    model_path = 'resnet18_quickdraw.onnx'
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model = ort.InferenceSession(os.path.join(dir_path, model_path))
    return model


@tensorleap_integration_test()
def check_custom_test_mapping(idx, subset):
    image = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)

    model = load_model()
    input_name = model.get_inputs()[0].name
    y_pred = model.run(None, {input_name: image})



    img_vis = image_visualizer(image)

    visualize(img_vis)

    metric_result = custom_metric_accuracy(gt, y_pred[0])

    loss_ret = custom_loss_categorical_crossentropy(gt, y_pred[0])

    m1 = metadata_country(idx, subset)
    m2 = metadata_label(idx, subset)
    m3 = metadata_recognized(idx, subset)

    # here the user can return whatever he wants


if __name__ == '__main__':
    check_custom_test_mapping(0, preprocess_func()[0])