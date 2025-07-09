class Pipeline():

    def __init__(self):
        pass
    
    def train_val_pred(self):
        from logger import logging

        from data_ingestion import DataIngestion
        DataIngestion()

        from data_preprocessing import DataPreprocessing
        dataPreprocessingObject = DataPreprocessing()
        dataPreprocessingObject.pre_processing()

        from training_prediction import TrainingPrediction
        trainingPredictionObject = TrainingPrediction(dataPreprocessingObject)

        trainingPredictionObject.training_and_validation()

        logging.info(f"Initiating Model Eval comparisons ..")

        from evaluation_comparison import Eval_comparison
        evalComparisonObject = Eval_comparison()
        evalComparisonObject.comp_btw_rnns()
        evalComparisonObject.comp_btw_models()

        logging.info(f"Model Eval comparisons Complete ..")

        logging.info(f"Initiating Model Feature Imp using ShAp ..")

        from feature_importance_shap import Feature_Imp
        featureImpObject = Feature_Imp()
        featureImpObject.explainer()

        logging.info(f"Model Feature Imp using ShAp Complete ..")

        logging.info(f"Hypothesis_testing ..")

        from hypothesis_testing import HypothesisTest
        hypothesisTestObject = HypothesisTest()

        hypothesisTestObject.hyp_test()

        logging.info(f"Hypothesis Testing Complete ..")