from ts.torch_handler.base_handler import BaseHandler
import os
import torch
import warnings
import json


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    #
    # def __init__(self):
    #     super().__init__()

    def __init__(self):
        self._context = None
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        self._context = context
        self.initialized = True
        properties = context.system_properties
        # print(properties)

        #  load the model
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        print('-----------------------1----------------------')
        print(model_dir)
        warnings.warn(model_dir)
        print('-----------------------1----------------------')

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        print('-----------------------2----------------------')
        print(model_pt_path)
        print('------------------------2---------------------')

        self.model = torch.load(model_pt_path)

        # self.model = torch.load(model_pt_path)

        self.initialized = True

        # ssdb_data = pyssdb.Client(host='localhost', port=20210)
        # # finish updating main model, switch to main model and backup main model
        # ssdb_data.set('main_model_state', 'serving')
        # ssdb_data.set('vice_model_state', 'copying')
        # print('finish switching flag')

    def preprocess(self, data: list):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        text = data[0].get("body")  # json_data(byte)

        if text is None:
            warnings.warn("data params is none")
            raise Exception("no data")
        else:
            data = json.loads(text)  # dict, keys containing 'user_list' and 'num_K'

        return data

    def inference(self, data):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        return [[1]]

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction,list containing key 'body' and value json_data
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """

        data = self.preprocess(data)
        # print(data)
        model_output = self.inference(data)
        # self.postprocess(model_output)

        return model_output


service = ModelHandler()


def handle(data, context):
    if not service.initialized:
        service.initialize(context)
    if data is None:
        return None
    return service.handle(data, context)
